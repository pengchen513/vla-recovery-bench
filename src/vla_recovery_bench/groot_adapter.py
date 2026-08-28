"""Frozen GR00T N1.5 adapter for the audited RoboCasa PandaOmron contract."""

from __future__ import annotations

import io
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from .robocasa_adapter import validate_action
from .types import Action, Observation

CAMERA_SHAPES = {
    "video.robot0_agentview_left": (256, 256, 3),
    "video.robot0_agentview_right": (256, 256, 3),
    "video.robot0_eye_in_hand": (256, 256, 3),
}
STATE_SHAPES = {
    "state.end_effector_position_relative": (3,),
    "state.end_effector_rotation_relative": (4,),
    "state.gripper_qpos": (2,),
    "state.base_position": (3,),
    "state.base_rotation": (4,),
}
PROMPT_KEY = "annotation.human.task_description"
ACTION_DIMS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}
ACTION_HORIZON = 16


def flatten_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested mappings while preserving every leaf and dotted key."""
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
            return
        if prefix in flattened:
            raise ValueError(f"duplicate observation key after flattening: {prefix}")
        flattened[prefix] = value

    visit("", observation)
    return flattened


def task_description(observation: Mapping[str, Any]) -> str:
    """Return the audited scalar task prompt without changing its contents."""
    value = flatten_observation(observation).get(PROMPT_KEY)
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"{PROMPT_KEY} must be a scalar or one-element array; got shape={array.shape}"
        )
    prompt = str(array.reshape(-1)[0])
    if not prompt:
        raise ValueError("RoboCasa task description is empty")
    return prompt


def prepare_groot_observation(
    observation: Mapping[str, Any], instruction: str
) -> dict[str, np.ndarray]:
    """Validate and add the single temporal dimension expected by GR00T."""
    flattened = flatten_observation(observation)
    expected = set(CAMERA_SHAPES) | set(STATE_SHAPES) | {PROMPT_KEY}
    if set(flattened) != expected:
        raise ValueError(
            "observation keys do not match audited GR00T contract: "
            f"missing={sorted(expected - set(flattened))}, "
            f"unexpected={sorted(set(flattened) - expected)}"
        )

    prepared: dict[str, np.ndarray] = {}
    for key, shape in CAMERA_SHAPES.items():
        value = np.asarray(flattened[key])
        if value.shape != shape or value.dtype != np.uint8:
            raise ValueError(
                f"camera contract mismatch for {key}: "
                f"expected shape={shape}, dtype=uint8; got shape={value.shape}, dtype={value.dtype}"
            )
        prepared[key] = np.expand_dims(value, axis=0)

    for key, shape in STATE_SHAPES.items():
        value = np.asarray(flattened[key])
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(
                f"state contract mismatch for {key}: "
                f"expected shape={shape}, floating dtype; "
                f"got shape={value.shape}, dtype={value.dtype}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"state contains NaN or Inf: {key}")
        prepared[key] = np.expand_dims(value, axis=0)

    prompt = task_description(observation)
    if instruction and instruction != prompt:
        raise ValueError(
            "instruction does not match annotation.human.task_description; "
            "the adapter will not silently replace the environment prompt"
        )
    prepared[PROMPT_KEY] = np.asarray([prompt])
    return prepared


class TorchZmqPolicyClient:
    """Small client compatible with GR00T's official torch/ZeroMQ service."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 120000):
        import torch
        import zmq

        self._torch = torch
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        output = io.BytesIO()
        self._torch.save(request, output)
        self._socket.send(output.getvalue())
        response = self._torch.load(io.BytesIO(self._socket.recv()), weights_only=False)
        if not isinstance(response, dict):
            raise RuntimeError(f"invalid policy response type: {type(response)}")
        if "error" in response:
            raise RuntimeError(f"policy server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            return self.call("ping") == {"status": "ok", "message": "Server is running"}
        except (RuntimeError, self._zmq.ZMQError):
            return False

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None


class GrootRoboCasaPolicy:
    """Execute audited 16-step GR00T chunks as structured RoboCasa actions."""

    name = "groot_n1_5_robocasa_atomic_seen_30p"
    action_chunk_length = ACTION_HORIZON

    def __init__(self, action_space: Any, client: Any) -> None:
        self.action_space = action_space
        self.client = client
        self._actions: deque[Action] = deque()
        self._requested_action_chunk: tuple[Action, ...] = ()
        self.last_inference_latency_ms: float | None = None
        self.last_chunk_saturation: dict[str, Any] | None = None
        self.inference_count = 0
        self.episode_saturated_values = 0

    def reset(self) -> None:
        self._actions.clear()
        self._requested_action_chunk = ()
        self.last_inference_latency_ms = None
        self.last_chunk_saturation = None
        self.episode_saturated_values = 0

    def act(self, observation: Observation, instruction: str) -> Action:
        self.last_inference_latency_ms = None
        self.last_chunk_saturation = None
        if not self._actions:
            model_observation = prepare_groot_observation(observation, instruction)
            started = time.perf_counter()
            response = self.client.call("get_action", model_observation)
            self.last_inference_latency_ms = (time.perf_counter() - started) * 1000.0
            self.inference_count += 1
            actions = self._validate_action_chunk(response)
            self._actions.extend(actions)
            self._requested_action_chunk = tuple(_copy_action(action) for action in actions)
        action = self._actions.popleft()
        validate_action(self.action_space, action)
        return action

    @property
    def requested_action_chunk(self) -> tuple[Action, ...]:
        """Return the complete chunk emitted by the latest policy query."""
        return tuple(_copy_action(action) for action in self._requested_action_chunk)

    def _validate_action_chunk(self, response: Mapping[str, Any]) -> list[Action]:
        if set(response) != set(ACTION_DIMS):
            raise ValueError(
                "GR00T action keys do not match audited contract: "
                f"expected={sorted(ACTION_DIMS)}, got={sorted(response)}"
            )
        arrays: dict[str, np.ndarray] = {}
        saturation: dict[str, Any] = {}
        saturated_values = 0
        for key, width in ACTION_DIMS.items():
            value = response[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.shape != (ACTION_HORIZON, width):
                raise ValueError(
                    f"GR00T action chunk mismatch for {key}: "
                    f"expected={(ACTION_HORIZON, width)}, got={array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"GR00T action chunk contains NaN or Inf: {key}")
            child_space = self.action_space.spaces[key]
            below = array < child_space.low
            above = array > child_space.high
            count = int(np.count_nonzero(below | above))
            saturation[key] = {
                "raw_minimum": float(array.min()),
                "raw_maximum": float(array.max()),
                "saturated_values": count,
            }
            saturated_values += count
            arrays[key] = np.clip(array, child_space.low, child_space.high).astype(
                np.float32, copy=False
            )

        self.last_chunk_saturation = {
            "fields": saturation,
            "saturated_values": saturated_values,
        }
        self.episode_saturated_values += saturated_values

        actions: list[Action] = []
        for index in range(ACTION_HORIZON):
            action = {key: value[index].copy() for key, value in arrays.items()}
            validate_action(self.action_space, action)
            actions.append(action)
        return actions


def _copy_action(action: Action) -> Action:
    if isinstance(action, Mapping):
        return {key: np.asarray(value).copy() for key, value in action.items()}
    return tuple(float(value) for value in action)
