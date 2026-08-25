from __future__ import annotations

from typing import Protocol

from .types import (
    Action,
    FaultApplication,
    FaultSpec,
    MonitorContext,
    MonitorDecision,
    Observation,
    RecoveryContext,
    RecoveryDecision,
    StepTransition,
)


class EnvironmentAdapter(Protocol):
    def reset(self, seed: int) -> Observation: ...

    def step(self, action: Action) -> StepTransition: ...

    def inject_fault(self, fault: FaultSpec) -> FaultApplication: ...

    def close(self) -> None: ...


class FrozenPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: Observation, instruction: str) -> Action: ...


class FailureMonitor(Protocol):
    def reset(self) -> None: ...

    def observe(self, context: MonitorContext) -> MonitorDecision: ...


class RecoveryController(Protocol):
    def reset(self) -> None: ...

    def decide(self, context: RecoveryContext) -> RecoveryDecision: ...

    def execute(self, decision: RecoveryDecision) -> None: ...
