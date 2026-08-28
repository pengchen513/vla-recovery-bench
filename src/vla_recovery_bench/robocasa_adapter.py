from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .types import Action, FaultApplication, FaultSpec, Observation, StepTransition

DEFAULT_POLICY_CAMERA = "video.robot0_agentview_left"


def _copy_array(value: Any) -> Any:
    import numpy as np

    return np.asarray(value).copy()


def _copy_action(action: Action) -> Action:
    if isinstance(action, Mapping):
        return {str(key): _copy_action(value) for key, value in action.items()}
    if isinstance(action, tuple):
        return tuple(_copy_action(value) for value in action)
    if isinstance(action, list):
        return [_copy_action(value) for value in action]
    return _copy_array(action)


def corrupt_image(image: Any, variant: str, stale_frame: Any | None = None) -> Any:
    """Apply a deterministic, shape-preserving observation corruption."""
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected HxWx3/HxWx4 image, got {array.shape}")
    if variant == "stale_frame":
        if stale_frame is None:
            raise ValueError("stale_frame corruption requires a cached prior image")
        stale = np.asarray(stale_frame)
        if stale.shape != array.shape or stale.dtype != array.dtype:
            raise ValueError("stale_frame does not match current image shape and dtype")
        return stale.copy()
    result = array.copy()
    if variant == "all_zero":
        result[...] = 0
        return result
    if variant == "partial_mask":
        height, width = result.shape[:2]
        row_start, row_stop = height // 4, (3 * height) // 4
        col_start, col_stop = width // 4, (3 * width) // 4
        result[row_start:row_stop, col_start:col_stop, :3] = 0
        return result
    if variant == "blur":
        rgb = result[..., :3].astype(np.float32)
        padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blurred = sum(
            padded[row : row + rgb.shape[0], col : col + rgb.shape[1]]
            for row in range(3)
            for col in range(3)
        ) / 9.0
        result[..., :3] = blurred.astype(result.dtype)
        return result
    if variant == "color_shift":
        rgb = result[..., :3].astype(np.float32)
        rgb *= np.asarray([0.55, 1.2, 0.7], dtype=np.float32)
        if np.issubdtype(result.dtype, np.integer):
            limits = np.iinfo(result.dtype)
            rgb = np.clip(rgb, limits.min, limits.max)
        else:
            rgb = np.clip(rgb, 0.0, 1.0)
        result[..., :3] = rgb.astype(result.dtype)
        return result
    raise ValueError(f"unsupported observation corruption: {variant}")


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
        self._actuator_steps = 0
        self._actuator_variant: str | None = None
        self._actuator_elapsed = 0
        self._fault_rng = None
        self._last_executed_action: Action | None = None
        self._occlusion_steps = 0
        self._occlusion_variant: str | None = None
        self._occlusion_keys: set[str] | None = None
        self._last_image_frames: dict[str, Any] = {}
        self._pre_step_image_frames: dict[str, Any] = {}

    @property
    def action_space(self) -> Any:
        """The unmodified Gymnasium action space exposed by RoboCasa."""
        return self._env.action_space

    @property
    def raw_environment(self) -> Any:
        return self._env

    def reset(self, seed: int) -> Observation:
        # Fault state is episode-local; never leak a pending fault into a new seed.
        self._dropout_steps = 0
        self._actuator_steps = 0
        self._actuator_variant = None
        self._actuator_elapsed = 0
        self._fault_rng = None
        self._last_executed_action = None
        self._occlusion_steps = 0
        self._occlusion_variant = None
        self._occlusion_keys = None
        self._pre_step_image_frames = {}
        observation, _ = self._env.reset(seed=seed)
        self._cache_image_frames(observation)
        return self._transform_observation(observation)

    def step(self, action: Action) -> StepTransition:
        validate_action(self.action_space, action)
        self._pre_step_image_frames = {
            key: _copy_array(value) for key, value in self._last_image_frames.items()
        }
        actual_action = action
        if self._dropout_steps > 0:
            actual_action = zero_action(self.action_space)
            self._dropout_steps -= 1
        elif self._actuator_steps > 0:
            actual_action = self._apply_actuator_variant(action)
            self._actuator_steps -= 1
            self._actuator_elapsed += 1
            if self._actuator_steps == 0:
                self._actuator_variant = None
                self._fault_rng = None
                self._actuator_elapsed = 0
        validate_action(self.action_space, actual_action)
        observation, reward, terminated, truncated, info = self._env.step(actual_action)
        self._last_executed_action = _copy_action(actual_action)
        return StepTransition(
            observation=self._transform_observation(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            executed_action=actual_action,
        )

    def apply_pending_observation_fault(self, observation: Observation) -> Observation:
        """Apply an after-step sensor fault to the observation just returned.

        The normal ``step`` path transforms observations before the runner can
        inject an ``after_step`` fault.  This hook applies the newly injected
        fault to that returned observation and consumes one duration step, so a
        schedule at ``onset - 1`` produces the first affected input at ``onset``.
        """
        if self._occlusion_steps <= 0:
            return observation
        if self._occlusion_variant == "stale_frame" and self._pre_step_image_frames:
            self._last_image_frames = {
                key: _copy_array(value) for key, value in self._pre_step_image_frames.items()
            }
            try:
                transformed = self._transform_observation(observation)
            finally:
                if self._occlusion_steps <= 0:
                    self._cache_image_frames(observation)
            return transformed
        transformed = self._transform_observation(observation)
        if self._occlusion_steps <= 0:
            self._cache_image_frames(observation)
        return transformed

    def inject_fault(self, fault: FaultSpec) -> FaultApplication:
        if fault.kind == "actuator_dropout":
            duration = int(fault.parameters.get("duration", 1))
            if duration <= 0:
                raise ValueError("actuator_dropout duration must be positive")
            self._dropout_steps = max(self._dropout_steps, duration)
            return FaultApplication(
                fault.fault_id, fault.kind, fault.step, True, {"duration": duration}
            )
        if fault.kind == "actuator_variant":
            import numpy as np

            duration = int(fault.parameters.get("duration", 1))
            variant = str(fault.parameters.get("variant", "all_channel_zero"))
            allowed = {
                "all_channel_zero",
                "arm_only",
                "base_only",
                "gripper_only",
                "hold_last",
                "intermittent",
                "bounded_noise",
            }
            if duration <= 0:
                raise ValueError("actuator_variant duration must be positive")
            if variant not in allowed:
                raise ValueError(f"unsupported actuator variant: {variant}")
            if variant in {"arm_only", "base_only", "gripper_only"}:
                required = {
                    "arm_only": {
                        "action.end_effector_position",
                        "action.end_effector_rotation",
                    },
                    "base_only": {"action.base_motion"},
                    "gripper_only": {"action.gripper_close"},
                }[variant]
                missing = required - set(self.action_space.spaces)
                if missing:
                    raise ValueError(
                        f"{variant} actuator variant requires action fields: {sorted(missing)}"
                    )
            self._actuator_steps = max(self._actuator_steps, duration)
            self._actuator_variant = variant
            self._actuator_elapsed = 0
            if variant == "bounded_noise":
                seed = int(fault.parameters.get("noise_seed", 0))
                self._fault_rng = np.random.default_rng(seed)
            return FaultApplication(
                fault.fault_id,
                fault.kind,
                fault.step,
                True,
                {
                    "duration": duration,
                    "variant": variant,
                    "first_affected_input_step": fault.step,
                },
            )
        if fault.kind == "observation_occlusion":
            duration = int(fault.parameters.get("duration", 1))
            if duration <= 0:
                raise ValueError("observation_occlusion duration must be positive")
            raw_keys = fault.parameters.get("camera_keys")
            if raw_keys is None:
                camera_key = fault.parameters.get("camera_key", DEFAULT_POLICY_CAMERA)
                raw_keys = [camera_key]
            if isinstance(raw_keys, str):
                raw_keys = [raw_keys]
            camera_keys = {str(key) for key in raw_keys}
            if not camera_keys:
                raise ValueError("observation_occlusion requires at least one camera key")
            self._occlusion_steps = max(self._occlusion_steps, duration)
            self._occlusion_variant = "all_zero"
            self._occlusion_keys = camera_keys
            return FaultApplication(
                fault.fault_id,
                fault.kind,
                fault.step,
                True,
                {
                    "duration": duration,
                    "variant": "all_zero",
                    "camera_keys": sorted(camera_keys),
                    "first_affected_input_step": fault.step + 1,
                },
            )
        if fault.kind == "observation_variant":
            duration = int(fault.parameters.get("duration", 1))
            variant = str(fault.parameters.get("variant", "partial_mask"))
            raw_keys = fault.parameters.get("camera_keys")
            if raw_keys is None:
                raw_keys = fault.parameters.get("camera_key", DEFAULT_POLICY_CAMERA)
            if isinstance(raw_keys, str):
                raw_keys = [raw_keys]
            camera_keys = {str(key) for key in raw_keys}
            allowed = {"partial_mask", "blur", "stale_frame", "color_shift", "all_zero"}
            if duration <= 0:
                raise ValueError("observation_variant duration must be positive")
            if variant not in allowed:
                raise ValueError(f"unsupported observation variant: {variant}")
            if not camera_keys:
                raise ValueError("observation_variant requires at least one camera key")
            known_keys = set(self._last_image_frames)
            missing = camera_keys - known_keys
            if missing:
                raise ValueError(
                    "observation_variant camera keys are not in the observation "
                    f"contract: {sorted(missing)}"
                )
            first_affected = int(
                fault.parameters.get("first_affected_input_step", fault.step + 1)
            )
            self._occlusion_steps = max(self._occlusion_steps, duration)
            self._occlusion_variant = variant
            self._occlusion_keys = camera_keys
            return FaultApplication(
                fault.fault_id,
                fault.kind,
                fault.step,
                True,
                {
                    "duration": duration,
                    "variant": variant,
                    "camera_keys": sorted(camera_keys),
                    "first_affected_input_step": first_affected,
                },
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
            self._cache_image_frames(observation)
            return observation
        variant = self._occlusion_variant or "all_zero"
        seen: set[str] = set()

        def mask(prefix: str, value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: mask(f"{prefix}.{key}" if prefix else str(key), item)
                    for key, item in value.items()
                }
            shape = getattr(value, "shape", ())
            if (
                prefix in (self._occlusion_keys or set())
                and len(shape) == 3
                and shape[-1] in (3, 4)
            ):
                seen.add(prefix)
                return corrupt_image(
                    value,
                    variant=variant,
                    stale_frame=self._last_image_frames.get(prefix),
                )
            return value

        transformed = mask("", observation)
        if seen != (self._occlusion_keys or set()):
            missing = (self._occlusion_keys or set()) - seen
            raise RuntimeError(f"observation fault did not affect camera keys: {sorted(missing)}")
        self._occlusion_steps -= 1
        if self._occlusion_steps == 0:
            self._occlusion_variant = None
            self._occlusion_keys = None
        return transformed

    def _cache_image_frames(self, observation: Observation) -> None:
        frames: dict[str, Any] = {}

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
                return
            shape = getattr(value, "shape", ())
            if len(shape) == 3 and shape[-1] in (3, 4):
                frames[prefix] = _copy_array(value)

        visit("", observation)
        self._last_image_frames = frames

    def _apply_actuator_variant(self, action: Action) -> Action:
        import numpy as np

        variant = self._actuator_variant or "all_channel_zero"
        if variant == "hold_last":
            if self._last_executed_action is not None:
                return _copy_action(self._last_executed_action)
            return zero_action(self.action_space)
        if variant == "intermittent" and self._actuator_elapsed % 2 == 1:
            return _copy_action(action)

        transformed = _copy_action(action)
        if not isinstance(transformed, Mapping):
            raise ValueError("RoboCasa actuator variants require a structured mapping action")
        zero_keys = {
            "all_channel_zero": set(self.action_space.spaces),
            "arm_only": {
                "action.end_effector_position",
                "action.end_effector_rotation",
            },
            "base_only": {"action.base_motion"},
            "gripper_only": {"action.gripper_close"},
            "intermittent": set(self.action_space.spaces),
        }.get(variant, set())
        if variant in {
            "all_channel_zero",
            "arm_only",
            "base_only",
            "gripper_only",
            "intermittent",
        }:
            for key in zero_keys:
                transformed[key] = np.zeros_like(transformed[key])
            return transformed
        if variant == "bounded_noise":
            if self._fault_rng is None:
                raise RuntimeError("bounded_noise actuator variant is missing its RNG")
            for key, child_space in self.action_space.spaces.items():
                value = np.asarray(transformed[key], dtype=np.float32)
                noise = self._fault_rng.normal(0.0, 0.05, size=value.shape)
                transformed[key] = np.clip(value + noise, child_space.low, child_space.high).astype(
                    child_space.dtype,
                    copy=False,
                )
            return transformed
        raise ValueError(f"unsupported active actuator variant: {variant}")

    def close(self) -> None:
        self._env.close()


def zero_action(action_space: Any) -> Action:
    """Construct a legal zero-like action while preserving space structure."""
    import numpy as np
    from gymnasium import spaces

    if isinstance(action_space, spaces.Box):
        value = np.zeros(action_space.shape, dtype=action_space.dtype)
        return np.clip(value, action_space.low, action_space.high)
    if isinstance(action_space, spaces.Discrete):
        return int(action_space.start)
    if isinstance(action_space, spaces.MultiDiscrete):
        return np.asarray(action_space.start, dtype=action_space.dtype).copy()
    if isinstance(action_space, spaces.MultiBinary):
        return np.zeros(action_space.shape, dtype=action_space.dtype)
    if isinstance(action_space, spaces.Dict):
        return {key: zero_action(child) for key, child in action_space.spaces.items()}
    if isinstance(action_space, spaces.Tuple):
        return tuple(zero_action(child) for child in action_space.spaces)
    return action_space.sample()


def validate_action(action_space: Any, action: Action) -> None:
    """Fail closed when a policy emits an invalid action."""
    import numpy as np
    from gymnasium import spaces

    if isinstance(action_space, spaces.Dict):
        if not isinstance(action, Mapping) or set(action) != set(action_space.spaces):
            raise ValueError("mapping action keys do not match RoboCasa action space")
        for key, child_space in action_space.spaces.items():
            validate_action(child_space, action[key])
        return
    if isinstance(action_space, spaces.Tuple):
        if not isinstance(action, tuple) or len(action) != len(action_space.spaces):
            raise ValueError("tuple action does not match action space")
        for child_space, child_action in zip(action_space.spaces, action, strict=True):
            validate_action(child_space, child_action)
        return
    array = np.asarray(action)
    if array.dtype.kind in "biufc" and not np.all(np.isfinite(array)):
        raise ValueError("policy action contains NaN or Inf")
    if not action_space.contains(action):
        raise ValueError(
            f"policy action is outside action space: shape={array.shape}, dtype={array.dtype}"
        )


def describe_action_space(action_space: Any) -> dict[str, Any]:
    """Serialize a Gymnasium action space without flattening structured fields."""
    import numpy as np

    result: dict[str, Any] = {
        "type": f"{type(action_space).__module__}.{type(action_space).__qualname__}",
        "shape": list(action_space.shape) if getattr(action_space, "shape", None) else None,
        "dtype": str(action_space.dtype) if getattr(action_space, "dtype", None) else None,
    }
    for name in ("low", "high", "n", "nvec", "start", "stop"):
        if not hasattr(action_space, name):
            continue
        value = getattr(action_space, name)
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, np.generic):
            result[name] = value.item()
        else:
            result[name] = value
    spaces = getattr(action_space, "spaces", None)
    if isinstance(spaces, Mapping):
        result["spaces"] = {
            str(key): describe_action_space(child) for key, child in spaces.items()
        }
    elif spaces is not None:
        result["spaces"] = [describe_action_space(child) for child in spaces]
    return result


def action_shape(action: Action) -> Any:
    """Return a JSON-friendly recursive action shape."""
    if isinstance(action, Mapping):
        return {str(key): action_shape(value) for key, value in action.items()}
    if isinstance(action, tuple):
        return [action_shape(value) for value in action]
    shape = getattr(action, "shape", None)
    return [int(dimension) for dimension in shape] if shape is not None else []


class ZeroPolicy:
    """Deterministic interface baseline; it has no trainable parameters."""

    name = "zero"

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def reset(self) -> None:
        return None

    def act(self, observation: Observation, instruction: str) -> Action:
        del observation, instruction
        action = zero_action(self.action_space)
        validate_action(self.action_space, action)
        return action


class RandomPolicy:
    """Random interface baseline; it is not a scientific policy result."""

    name = "random"

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def reset(self) -> None:
        return None

    def act(self, observation: Observation, instruction: str) -> Action:
        del observation, instruction
        action = self.action_space.sample()
        validate_action(self.action_space, action)
        return action
