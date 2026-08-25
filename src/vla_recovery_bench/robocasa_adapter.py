from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .types import Action, FaultApplication, FaultSpec, Observation, StepTransition


def find_rgb_observations(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Find HxWx3 image arrays in a possibly nested observation mapping."""
    images: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), item)
            return
        shape = getattr(value, "shape", ())
        if len(shape) == 3 and shape[-1] in (3, 4):
            images[prefix] = value[..., :3]

    visit("", observation)
    return images


def save_rgb_image(image: Any, output_path: str | Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got shape {array.shape}")
    if array.dtype != np.uint8:
        if float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    # MuJoCo camera observations may be vertically flipped.
    array = np.flipud(array)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    return {
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "minimum": int(array.min()),
        "maximum": int(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
    }


class RoboCasaEnvironment:
    """Thin adapter around a Gymnasium RoboCasa environment.

    Faults that can be implemented without task-specific MuJoCo body names live
    here. Physical object displacement belongs in a task adapter and is rejected
    until its target body and coordinate frame are explicit.
    """

    def __init__(self, environment_id: str, split: str = "target", **kwargs: Any) -> None:
        import gymnasium as gym
        import robocasa  # noqa: F401

        self._env = gym.make(environment_id, split=split, **kwargs)
        self._dropout_steps = 0
        self._occlusion_steps = 0

    @property
    def raw_environment(self) -> Any:
        return self._env

    def reset(self, seed: int) -> Observation:
        observation, _ = self._env.reset(seed=seed)
        return self._transform_observation(observation)

    def step(self, action: Action) -> StepTransition:
        import numpy as np

        actual_action = np.asarray(action, dtype=np.float32)
        if self._dropout_steps > 0:
            actual_action = np.zeros_like(actual_action)
            self._dropout_steps -= 1
        observation, reward, terminated, truncated, info = self._env.step(actual_action)
        return StepTransition(
            observation=self._transform_observation(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def inject_fault(self, fault: FaultSpec) -> FaultApplication:
        if fault.kind == "actuator_dropout":
            duration = int(fault.parameters.get("duration", 1))
            self._dropout_steps = max(self._dropout_steps, duration)
            return FaultApplication(
                fault.fault_id, fault.kind, fault.step, True, {"duration": duration}
            )
        if fault.kind == "observation_occlusion":
            duration = int(fault.parameters.get("duration", 1))
            self._occlusion_steps = max(self._occlusion_steps, duration)
            return FaultApplication(
                fault.fault_id, fault.kind, fault.step, True, {"duration": duration}
            )
        return FaultApplication(
            fault.fault_id,
            fault.kind,
            fault.step,
            False,
            {"reason": "fault requires a task-specific RoboCasa adapter"},
        )

    def _transform_observation(self, observation: Observation) -> Observation:
        if self._occlusion_steps <= 0:
            return observation
        import numpy as np

        def mask(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {key: mask(item) for key, item in value.items()}
            shape = getattr(value, "shape", ())
            if len(shape) == 3 and shape[-1] in (3, 4):
                return np.zeros_like(value)
            return value

        transformed = mask(observation)
        self._occlusion_steps -= 1
        return transformed

    def close(self) -> None:
        self._env.close()
