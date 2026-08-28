from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any

Observation = Mapping[str, Any]
# RoboCasa uses a Gymnasium Dict action space while the dummy environment uses
# a short numeric sequence. Keep both representations explicit at the boundary.
Action = Mapping[str, Any] | Sequence[float]


class FaultPhase(StrEnum):
    BEFORE_ACTION = "before_action"
    AFTER_STEP = "after_step"


class RecoveryAction(StrEnum):
    CONTINUE = "continue"
    REQUERY_POLICY = "requery_policy"
    REISSUE_CURRENT_CHUNK = "reissue_current_chunk"
    SWITCH_CAMERA_SUBSET = "switch_camera_subset"
    DIAGNOSTIC_PROBE = "diagnostic_probe"
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
    # Adapters may expose the action that actually reached the environment for
    # the offline audit stream. It is deliberately absent from MonitorContext.
    executed_action: Action | None = None


@dataclass(frozen=True)
class AuditRecord:
    """Privileged transition data kept outside the monitor input boundary."""

    episode_id: int
    step: int
    requested_action: Action
    executed_action: Action | None
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    success: bool = False


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
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, RecoveryAction):
            raise TypeError("recovery action must be a RecoveryAction enum value")
        if not self.reason:
            raise ValueError("recovery decision reason must not be empty")


_FORBIDDEN_MONITOR_KEYS = frozenset(
    {
        "info",
        "reward",
        "terminated",
        "truncated",
        "fault",
        "schedule",
        "faultspec",
        "faultapplication",
        "fault_id",
        "fault_type",
        "fault_kind",
        "scheduled_fault_time",
        "executed_action",
        "success",
        "terminal_success",
        "mujoco_state",
        "body_pose",
        "body_id",
        "contact",
        "joint_torque",
        "dropout_steps",
        "occlusion_steps",
    }
)


def _assert_monitor_safe(value: Any, path: str = "observation") -> None:
    """Fail closed when a declared observation contains a privileged field."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            components = {
                component.lower().replace("-", "_")
                for component in key_text.replace("/", ".").split(".")
            }
            if components & _FORBIDDEN_MONITOR_KEYS:
                raise ValueError(f"forbidden monitor field at {path}.{key_text}")
            _assert_monitor_safe(child, f"{path}.{key_text}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_monitor_safe(child, f"{path}[{index}]")
        return
    if is_dataclass(value) and not isinstance(value, type):
        _assert_monitor_safe(vars(value), path)
        return
    # Arrays and scalar leaves are intentionally not coerced or copied here.
    # The adapter owns observation-shape and dtype validation.


@dataclass(frozen=True)
class ActionChunkMetadata:
    """Non-privileged metadata needed to reason about action-chunk timing."""

    chunk_id: int = 0
    position_in_chunk: int = 0
    chunk_length: int = 1
    remaining_horizon: int = 0
    policy_inference_latency_ms: float | None = None
    camera_keys: tuple[str, ...] = ()
    probe_active: bool = False

    def __post_init__(self) -> None:
        if self.chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        if not 0 <= self.position_in_chunk < self.chunk_length:
            raise ValueError("position_in_chunk must be within chunk_length")
        if self.remaining_horizon < 0:
            raise ValueError("remaining_horizon must be non-negative")
        if self.policy_inference_latency_ms is not None and self.policy_inference_latency_ms < 0:
            raise ValueError("policy_inference_latency_ms must be non-negative")


@dataclass(frozen=True)
class MonitorContext:
    episode_id: int
    step: int
    instruction: str
    previous_observation: Observation
    observation: Observation
    action: Action
    action_chunk: Sequence[Action] = ()
    chunk: ActionChunkMetadata = field(default_factory=ActionChunkMetadata)

    def __post_init__(self) -> None:
        # This is a structural firewall for the in-process runner. A separate
        # policy/monitor process is still recommended for adversarial isolation.
        _assert_monitor_safe(self.previous_observation, "previous_observation")
        _assert_monitor_safe(self.observation, "observation")
        _assert_monitor_safe(self.action, "requested_action")
        for index, action in enumerate(self.action_chunk):
            _assert_monitor_safe(action, f"requested_action_chunk[{index}]")

    @property
    def requested_action(self) -> Action:
        """Alias that makes the non-executed nature of the action explicit."""
        return self.action


@dataclass(frozen=True)
class RecoveryContext:
    episode_id: int
    step: int
    instruction: str
    observation: Observation
    monitor: MonitorDecision
    retry_count: int
    chunk: ActionChunkMetadata = field(default_factory=ActionChunkMetadata)

    def __post_init__(self) -> None:
        _assert_monitor_safe(self.observation, "recovery_observation")


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
    pair_id: str | None = None
    task_id: str | None = None
    split: str | None = None
    wall_time_seconds: float = 0.0
    policy_inference_latency_ms: tuple[float, ...] = ()
    intervention_actions: tuple[str, ...] = ()
