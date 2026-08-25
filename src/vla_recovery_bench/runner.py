from __future__ import annotations

from dataclasses import dataclass

from .faults import FaultSchedule
from .interfaces import EnvironmentAdapter, FailureMonitor, FrozenPolicy, RecoveryController
from .metrics import aggregate_episode_metrics
from .recording import JsonlRecorder
from .types import (
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

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.detection_match_window < 0:
            raise ValueError("detection_match_window must be non-negative")


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
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.monitor = monitor
        self.recovery = recovery
        self.fault_schedule = fault_schedule
        self.recorder = recorder
        self.config = config

    def run(self) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        self.recorder.record(
            "experiment_start",
            runner_config=self.config,
            fault_schedule=self.fault_schedule.faults,
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
        self.fault_schedule.reset()
        self.policy.reset()
        self.monitor.reset()
        self.recovery.reset()
        observation = self.environment.reset(seed)
        total_reward = 0.0
        applied_faults: dict[str, int] = {}
        detected_faults: set[str] = set()
        detection_delays: list[int] = []
        false_alarms = 0
        retry_count = 0
        recovery_attempted = False
        success = False
        termination_reason = "horizon"

        self.recorder.record("episode_start", episode_id=episode_id, seed=seed)

        for step in range(self.config.horizon):
            self._apply_due_faults(
                episode_id, step, FaultPhase.BEFORE_ACTION, applied_faults
            )
            action = self.policy.act(observation, self.config.instruction)
            transition = self.environment.step(action)
            total_reward += transition.reward
            self._apply_due_faults(episode_id, step, FaultPhase.AFTER_STEP, applied_faults)

            monitor_decision = self.monitor.observe(
                MonitorContext(
                    episode_id=episode_id,
                    step=step,
                    instruction=self.config.instruction,
                    previous_observation=observation,
                    observation=transition.observation,
                    action=action,
                    reward=transition.reward,
                    info=transition.info,
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
                )
            )
            if recovery_decision.action is not RecoveryAction.CONTINUE:
                recovery_attempted = True
                retry_count += 1
                if recovery_decision.action in {RecoveryAction.RETRY, RecoveryAction.REPLAN}:
                    # Stateful action-chunk policies must not continue a stale queue after recovery.
                    self.policy.reset()
                self.recovery.execute(recovery_decision)

            self.recorder.record(
                "step",
                episode_id=episode_id,
                step=step,
                action=action,
                reward=transition.reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                info=transition.info,
                monitor=monitor_decision,
                matched_fault_id=matched_fault_id,
                recovery=recovery_decision,
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
        )
        self.recorder.record("episode_end", result=result)
        return result

    def _apply_due_faults(
        self,
        episode_id: int,
        step: int,
        phase: FaultPhase,
        applied_faults: dict[str, int],
    ) -> None:
        for fault in self.fault_schedule.due(step, phase):
            application = self.environment.inject_fault(fault)
            if application.applied:
                applied_faults[fault.fault_id] = step
            self.recorder.record(
                "fault_injection",
                episode_id=episode_id,
                step=step,
                phase=phase,
                fault=fault,
                application=application,
            )
