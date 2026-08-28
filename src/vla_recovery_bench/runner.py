from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from .faults import FaultSchedule
from .interfaces import EnvironmentAdapter, FailureMonitor, FrozenPolicy, RecoveryController
from .metrics import aggregate_episode_metrics
from .recording import JsonlRecorder
from .types import (
    ActionChunkMetadata,
    AuditRecord,
    EpisodeResult,
    FaultPhase,
    MonitorContext,
    RecoveryAction,
    RecoveryContext,
)


@dataclass(frozen=True)
class RunnerConfig:
    episodes: int
    horizon: int
    base_seed: int
    instruction: str
    detection_match_window: int = 5
    pair_id_prefix: str | None = None
    task_id: str | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.detection_match_window < 0:
            raise ValueError("detection_match_window must be non-negative")
        if self.pair_id_prefix is not None and not self.pair_id_prefix:
            raise ValueError("pair_id_prefix must not be empty")


class ExperimentRunner:
    def __init__(
        self,
        environment: EnvironmentAdapter,
        policy: FrozenPolicy,
        monitor: FailureMonitor,
        recovery: RecoveryController,
        fault_schedule: FaultSchedule,
        recorder: JsonlRecorder,
        config: RunnerConfig,
        fault_schedule_factory: Callable[[int, int], FaultSchedule] | None = None,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.monitor = monitor
        self.recovery = recovery
        self.fault_schedule = fault_schedule
        self.recorder = recorder
        self.config = config
        self.fault_schedule_factory = fault_schedule_factory

    def run(self) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        self.recorder.record(
            "experiment_start",
            runner_config=self.config,
            fault_schedule=(
                self.fault_schedule.faults
                if self.fault_schedule_factory is None
                else "per_episode_schedule_factory"
            ),
        )
        try:
            for episode_id in range(self.config.episodes):
                results.append(self._run_episode(episode_id))
            self.recorder.record(
                "experiment_end",
                status="completed",
                metrics=aggregate_episode_metrics(results),
            )
        except Exception as error:
            self.recorder.record(
                "experiment_end",
                status="failed",
                completed_episodes=len(results),
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            self.environment.close()
        return results

    def _run_episode(self, episode_id: int) -> EpisodeResult:
        seed = self.config.base_seed + episode_id
        schedule = (
            self.fault_schedule
            if self.fault_schedule_factory is None
            else self.fault_schedule_factory(episode_id, seed)
        )
        if not isinstance(schedule, FaultSchedule):
            raise TypeError("fault_schedule_factory must return a FaultSchedule")
        schedule.reset()
        self.policy.reset()
        self.monitor.reset()
        self.recovery.reset()
        observation = self.environment.reset(seed)
        total_reward = 0.0
        # Values are the first policy-input step affected by each applied fault,
        # not necessarily the step at which an after-step fault was injected.
        applied_faults: dict[str, int] = {}
        detected_faults: set[str] = set()
        detection_delays: list[int] = []
        false_alarms = 0
        retry_count = 0
        recovery_attempted = False
        success = False
        termination_reason = "horizon"
        chunk_id = 0
        chunk_position = 0
        chunk_length = max(1, int(getattr(self.policy, "action_chunk_length", 1)))
        requested_action_history: list[object] = []
        policy_latencies: list[float] = []
        intervention_actions: list[str] = []
        episode_started = time.perf_counter()

        pair_id = (
            f"{self.config.pair_id_prefix}-{seed}"
            if self.config.pair_id_prefix is not None
            else None
        )
        self.recorder.record(
            "episode_start",
            episode_id=episode_id,
            episode_id_string=(f"{pair_id}-{episode_id}" if pair_id else None),
            pair_id=pair_id,
            seed=seed,
            task_id=self.config.task_id,
            split=self.config.split,
        )

        for step in range(self.config.horizon):
            self._apply_due_faults(
                schedule, episode_id, step, FaultPhase.BEFORE_ACTION, applied_faults
            )
            action = self.policy.act(observation, self.config.instruction)
            latency = getattr(self.policy, "last_inference_latency_ms", None)
            if latency is not None:
                policy_latencies.append(float(latency))
            transition = self.environment.step(action)
            total_reward += transition.reward
            after_fault_applied = self._apply_due_faults(
                schedule, episode_id, step, FaultPhase.AFTER_STEP, applied_faults
            )
            if after_fault_applied:
                apply_observation_fault = getattr(
                    self.environment, "apply_pending_observation_fault", None
                )
                if callable(apply_observation_fault):
                    transition = replace(
                        transition,
                        observation=apply_observation_fault(transition.observation),
                    )

            # The monitor receives requested actions and timing metadata only.
            # The executed action and simulator outcome are written separately
            # to the offline audit stream below.
            requested_action_history.append(action)
            history_window = tuple(requested_action_history[-chunk_length:])
            requested_action_chunk = getattr(self.policy, "requested_action_chunk", None)
            if requested_action_chunk is None:
                requested_action_chunk = history_window
            else:
                requested_action_chunk = tuple(requested_action_chunk)
            chunk = ActionChunkMetadata(
                chunk_id=chunk_id,
                position_in_chunk=chunk_position,
                chunk_length=chunk_length,
                remaining_horizon=max(self.config.horizon - step - 1, 0),
                policy_inference_latency_ms=getattr(self.policy, "last_inference_latency_ms", None),
            )

            monitor_decision = self.monitor.observe(
                MonitorContext(
                    episode_id=episode_id,
                    step=step,
                    instruction=self.config.instruction,
                    previous_observation=observation,
                    observation=transition.observation,
                    action=action,
                    action_chunk=requested_action_chunk,
                    chunk=chunk,
                )
            )

            matched_fault_id = None
            if monitor_decision.failure_detected:
                unmatched = [
                    (fault_id, injection_step)
                    for fault_id, injection_step in applied_faults.items()
                    if (
                        fault_id not in detected_faults
                        and injection_step <= step
                        and step - injection_step <= self.config.detection_match_window
                    )
                ]
                if unmatched:
                    matched_fault_id, injection_step = min(unmatched, key=lambda item: item[1])
                    detected_faults.add(matched_fault_id)
                    detection_delays.append(step - injection_step)
                else:
                    false_alarms += 1

            recovery_decision = self.recovery.decide(
                RecoveryContext(
                    episode_id=episode_id,
                    step=step,
                    instruction=self.config.instruction,
                    observation=transition.observation,
                    monitor=monitor_decision,
                    retry_count=retry_count,
                    chunk=chunk,
                )
            )
            if recovery_decision.action is not RecoveryAction.CONTINUE:
                recovery_attempted = True
                retry_count += 1
                intervention_actions.append(recovery_decision.action.value)
                if recovery_decision.action in {
                    RecoveryAction.REQUERY_POLICY,
                    RecoveryAction.REISSUE_CURRENT_CHUNK,
                    RecoveryAction.RETRY,
                    RecoveryAction.REPLAN,
                }:
                    # Stateful action-chunk policies must not continue a stale queue after recovery.
                    self.policy.reset()
                    chunk_id += 1
                    chunk_position = 0
                    requested_action_history.clear()
                elif chunk_position + 1 >= chunk_length:
                    chunk_id += 1
                    chunk_position = 0
                else:
                    chunk_position += 1
                self.recovery.execute(recovery_decision)
            elif chunk_position + 1 >= chunk_length:
                chunk_id += 1
                chunk_position = 0
            else:
                chunk_position += 1

            self.recorder.record(
                "step",
                episode_id=episode_id,
                step=step,
                requested_action=action,
                monitor=monitor_decision,
                matched_fault_id=matched_fault_id,
                recovery=recovery_decision,
                chunk=chunk,
                pair_id=pair_id,
                task_id=self.config.task_id,
                split=self.config.split,
            )
            self.recorder.record(
                "audit_transition",
                audit=AuditRecord(
                    episode_id=episode_id,
                    step=step,
                    requested_action=action,
                    executed_action=transition.executed_action,
                    reward=transition.reward,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    info=transition.info,
                    success=bool(transition.info.get("success", False)),
                ),
            )

            observation = transition.observation
            success = bool(transition.info.get("success", False))
            if success:
                termination_reason = "success"
                steps_taken = step + 1
                break
            if recovery_decision.action in {
                RecoveryAction.REQUEST_HELP,
                RecoveryAction.TERMINATE,
            }:
                termination_reason = recovery_decision.action.value
                steps_taken = step + 1
                break
            if transition.terminated or transition.truncated:
                termination_reason = "environment_done"
                steps_taken = step + 1
                break
        else:
            steps_taken = self.config.horizon

        fault_count = len(applied_faults)
        result = EpisodeResult(
            episode_id=episode_id,
            seed=seed,
            success=success,
            steps=steps_taken,
            reward=total_reward,
            fault_count=fault_count,
            detected_fault_count=len(detected_faults),
            false_alarm_count=false_alarms,
            detection_delays=tuple(detection_delays),
            recovery_attempted=recovery_attempted,
            recovery_success=bool(success and fault_count and recovery_attempted),
            termination_reason=termination_reason,
            pair_id=pair_id,
            task_id=self.config.task_id,
            split=self.config.split,
            wall_time_seconds=time.perf_counter() - episode_started,
            policy_inference_latency_ms=tuple(policy_latencies),
            intervention_actions=tuple(intervention_actions),
        )
        self.recorder.record("episode_end", result=result)
        return result

    def _apply_due_faults(
        self,
        schedule: FaultSchedule,
        episode_id: int,
        step: int,
        phase: FaultPhase,
        applied_faults: dict[str, int],
    ) -> bool:
        applied_any = False
        for fault in schedule.due(step, phase):
            application = self.environment.inject_fault(fault)
            if application.applied:
                applied_any = True
                first_affected_step = step + (1 if phase is FaultPhase.AFTER_STEP else 0)
                declared_first_affected = application.details.get("first_affected_input_step")
                if declared_first_affected is not None:
                    first_affected_step = int(declared_first_affected)
                applied_faults[fault.fault_id] = first_affected_step
            self.recorder.record(
                "fault_injection",
                episode_id=episode_id,
                step=step,
                phase=phase,
                fault=fault,
                application=application,
                injection_step=step,
                first_affected_input_step=(
                    (
                        int(application.details["first_affected_input_step"])
                        if "first_affected_input_step" in application.details
                        else step + (1 if phase is FaultPhase.AFTER_STEP else 0)
                    )
                    if application.applied
                    else None
                ),
            )
        return applied_any
