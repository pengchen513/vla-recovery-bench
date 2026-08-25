from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

Observation = Mapping[str, Any]
Action = Sequence[float]


class FaultPhase(StrEnum):
    BEFORE_ACTION = "before_action"
    AFTER_STEP = "after_step"


class RecoveryAction(StrEnum):
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    REQUEST_HELP = "request_help"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class FaultSpec:
    fault_id: str
    kind: str
    step: int
    phase: FaultPhase
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fault_id:
            raise ValueError("fault_id must not be empty")
        if self.step < 0:
            raise ValueError("fault step must be non-negative")


@dataclass(frozen=True)
class FaultApplication:
    fault_id: str
    kind: str
    requested_step: int
    applied: bool
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepTransition:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitorDecision:
    failure_detected: bool
    confidence: float
    failure_type: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str


@dataclass(frozen=True)
class MonitorContext:
    episode_id: int
    step: int
    instruction: str
    previous_observation: Observation
    observation: Observation
    action: Action
    reward: float
    info: Mapping[str, Any]


@dataclass(frozen=True)
class RecoveryContext:
    episode_id: int
    step: int
    instruction: str
    observation: Observation
    monitor: MonitorDecision
    retry_count: int


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: int
    seed: int
    success: bool
    steps: int
    reward: float
    fault_count: int
    detected_fault_count: int
    false_alarm_count: int
    detection_delays: tuple[int, ...]
    recovery_attempted: bool
    recovery_success: bool
    termination_reason: str
