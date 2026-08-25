from __future__ import annotations

from typing import Any

from .types import (
    Action,
    FaultApplication,
    FaultSpec,
    MonitorContext,
    MonitorDecision,
    Observation,
    RecoveryAction,
    RecoveryContext,
    RecoveryDecision,
    StepTransition,
)


class DummyProgressEnvironment:
    """Small deterministic environment used to test the experiment machinery."""

    def __init__(self, goal_progress: int = 8) -> None:
        self.goal_progress = goal_progress
        self.progress = 0.0
        self.dropout_steps = 0
        self.closed = False

    def _observation(self) -> Observation:
        return {
            "progress": self.progress,
            "goal_progress": self.goal_progress,
            "dropout_steps": self.dropout_steps,
        }

    def reset(self, seed: int) -> Observation:
        self.progress = 0.0
        self.dropout_steps = 0
        return self._observation()

    def step(self, action: Action) -> StepTransition:
        increment = float(action[0])
        if self.dropout_steps > 0:
            increment = 0.0
            self.dropout_steps -= 1
        self.progress += increment
        success = self.progress >= self.goal_progress
        return StepTransition(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            terminated=success,
            truncated=False,
            info={"success": success},
        )

    def inject_fault(self, fault: FaultSpec) -> FaultApplication:
        if fault.kind == "object_displacement":
            magnitude = float(fault.parameters.get("magnitude", 1.0))
            self.progress = max(0.0, self.progress - magnitude)
            return FaultApplication(
                fault_id=fault.fault_id,
                kind=fault.kind,
                requested_step=fault.step,
                applied=True,
                details={"magnitude": magnitude, "progress": self.progress},
            )
        if fault.kind == "actuator_dropout":
            duration = int(fault.parameters.get("duration", 1))
            self.dropout_steps = max(self.dropout_steps, duration)
            return FaultApplication(
                fault_id=fault.fault_id,
                kind=fault.kind,
                requested_step=fault.step,
                applied=True,
                details={"duration": duration},
            )
        return FaultApplication(
            fault_id=fault.fault_id,
            kind=fault.kind,
            requested_step=fault.step,
            applied=False,
            details={"reason": "unsupported dummy fault"},
        )

    def close(self) -> None:
        self.closed = True


class ConstantFrozenPolicy:
    def __init__(self, increment: float = 1.0) -> None:
        self.increment = increment

    def reset(self) -> None:
        return None

    def act(self, observation: Observation, instruction: str) -> Action:
        return [self.increment]


class ProgressFailureMonitor:
    def __init__(self, minimum_progress_delta: float = 0.5, stagnation_patience: int = 1) -> None:
        self.minimum_progress_delta = minimum_progress_delta
        self.stagnation_patience = stagnation_patience
        self.reset()

    def reset(self) -> None:
        self.stagnant_steps = 0

    def observe(self, context: MonitorContext) -> MonitorDecision:
        before = float(context.previous_observation["progress"])
        after = float(context.observation["progress"])
        delta = after - before
        if delta < 0:
            self.stagnant_steps = 0
            return MonitorDecision(
                failure_detected=True,
                confidence=1.0,
                failure_type="regression",
                evidence={"progress_delta": delta},
            )
        if delta < self.minimum_progress_delta:
            self.stagnant_steps += 1
        else:
            self.stagnant_steps = 0
        detected = self.stagnant_steps >= self.stagnation_patience
        return MonitorDecision(
            failure_detected=detected,
            confidence=0.9 if detected else 0.1,
            failure_type="stagnation" if detected else None,
            evidence={"progress_delta": delta, "stagnant_steps": self.stagnant_steps},
        )


class RetryRecoveryController:
    def __init__(self, max_retries: int = 4) -> None:
        self.max_retries = max_retries

    def reset(self) -> None:
        return None

    def execute(self, decision: RecoveryDecision) -> None:
        return None

    def decide(self, context: RecoveryContext) -> RecoveryDecision:
        if not context.monitor.failure_detected:
            return RecoveryDecision(RecoveryAction.CONTINUE, "no failure detected")
        if context.retry_count >= self.max_retries:
            return RecoveryDecision(RecoveryAction.REQUEST_HELP, "retry budget exhausted")
        return RecoveryDecision(RecoveryAction.RETRY, "retry after detected failure")


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value
