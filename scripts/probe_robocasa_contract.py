#!/usr/bin/env python3
"""Record the RoboCasa observation/action contract without loading a policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _path_key(prefix: str, key: Any) -> str:
    return f"{prefix}.{key}" if prefix else str(key)


def _json_number(value: Any) -> int | float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) <= 2**53:
        return int(number)
    return number


def _array_summary(
    value: Any,
) -> tuple[list[int], str | None, int | float | None, int | float | None]:
    """Summarize a leaf without changing the observation object."""
    import numpy as np

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        array = np.asarray(value)
        shape = array.shape
        dtype = array.dtype
    shape_list = [int(dimension) for dimension in shape]
    dtype_name = str(dtype)
    try:
        array = np.asarray(value)
        if array.size == 0 or not np.issubdtype(array.dtype, np.number):
            return shape_list, dtype_name, None, None
        return (
            shape_list,
            dtype_name,
            _json_number(np.min(array)),
            _json_number(np.max(array)),
        )
    except (TypeError, ValueError):
        return shape_list, dtype_name, None, None


def describe_observation(observation: Any) -> list[dict[str, Any]]:
    """Return one JSON-safe contract record for every observation leaf."""
    records: list[dict[str, Any]] = []

    def visit(path: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(_path_key(path, key), child)
            return
        shape, dtype, minimum, maximum = _array_summary(value)
        records.append(
            {
                "key": path,
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "shape": shape,
                "dtype": dtype,
                "minimum": minimum,
                "maximum": maximum,
            }
        )

    visit("", observation)
    return records


def find_contract_images(observation: Any) -> dict[str, Any]:
    """Find every HxWx3/HxWx4 image leaf, retaining the original value."""
    images: dict[str, Any] = {}

    def visit(path: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(_path_key(path, key), child)
            return
        shape = getattr(value, "shape", ())
        if len(shape) == 3 and int(shape[-1]) in (3, 4):
            images[path] = value

    visit("", observation)
    return images


def describe_action_space(action_space: Any) -> dict[str, Any]:
    """Describe a Gymnasium action space using JSON-safe values."""
    import numpy as np

    result: dict[str, Any] = {
        "type": f"{type(action_space).__module__}.{type(action_space).__qualname__}"
    }
    shape = getattr(action_space, "shape", None)
    result["shape"] = list(shape) if shape is not None else None
    dtype = getattr(action_space, "dtype", None)
    result["dtype"] = str(dtype) if dtype is not None else None
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
    if spaces is not None:
        if isinstance(spaces, Mapping):
            result["spaces"] = {
                str(key): describe_action_space(child) for key, child in spaces.items()
            }
        else:
            result["spaces"] = [describe_action_space(child) for child in spaces]
    return result


class ZeroPolicy:
    """Read-only interface baseline: choose a legal zero-like action."""

    name = "zero"

    def __call__(self, action_space: Any) -> Any:
        import numpy as np
        from gymnasium import spaces

        if isinstance(action_space, spaces.Box):
            action = np.zeros(action_space.shape, dtype=action_space.dtype)
            return np.clip(action, action_space.low, action_space.high)
        if isinstance(action_space, spaces.Discrete):
            return int(action_space.start)
        if isinstance(action_space, spaces.MultiDiscrete):
            return np.asarray(action_space.start, dtype=action_space.dtype).copy()
        if isinstance(action_space, spaces.MultiBinary):
            return np.zeros(action_space.shape, dtype=action_space.dtype)
        if isinstance(action_space, spaces.Dict):
            return {key: self(child) for key, child in action_space.spaces.items()}
        if isinstance(action_space, spaces.Tuple):
            return tuple(self(child) for child in action_space.spaces)
        return action_space.sample()


def validate_action(action_space: Any, action: Any) -> None:
    import numpy as np

    if isinstance(action, Mapping):
        if not isinstance(getattr(action_space, "spaces", None), Mapping):
            raise ValueError("mapping action does not match non-mapping action space")
        if set(action) != set(action_space.spaces):
            raise ValueError("mapping action keys do not match action space")
        for key, child_space in action_space.spaces.items():
            validate_action(child_space, action[key])
        return
    if isinstance(action, tuple):
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


def action_to_json(action: Any) -> Any:
    if isinstance(action, Mapping):
        return {str(key): action_to_json(value) for key, value in action.items()}
    if isinstance(action, tuple):
        return [action_to_json(value) for value in action]
    return _json_info(action)


def action_shape(action: Any) -> Any:
    if isinstance(action, Mapping):
        return {str(key): action_shape(value) for key, value in action.items()}
    if isinstance(action, tuple):
        return [action_shape(value) for value in action]
    shape = getattr(action, "shape", None)
    return list(shape) if shape is not None else []


def _json_info(value: Any) -> Any:
    """Convert info values to JSON without changing the live observation."""
    import numpy as np

    if isinstance(value, Mapping):
        return {str(key): _json_info(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _save_image(image: Any, path: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(array), mode="RGB").save(path)
    return {
        "path": str(path),
        "shape": [int(dimension) for dimension in array.shape],
        "dtype": str(array.dtype),
        "minimum": int(array.min()),
        "maximum": int(array.max()),
        "standard_deviation": float(array.std()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="/home/pc/VLA/outputs/robocasa_contract.json")
    parser.add_argument("--image-dir", default="/home/pc/VLA/outputs/contract_images")
    return parser.parse_args()


def main() -> int:
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("source /home/pc/VLA/env.sh before running the probe")

    import gymnasium
    import mujoco
    import robocasa
    import robosuite

    args = parse_args()
    output_path = Path(args.output)
    image_dir = Path(args.image_dir)
    if output_path.exists() and output_path.stat().st_size > 0:
        raise FileExistsError(f"refusing to overwrite non-empty artifact: {output_path}")
    if image_dir.exists() and any(image_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty image directory: {image_dir}")
    env = gymnasium.make(args.environment_id, split=args.split)
    policy = ZeroPolicy()
    try:
        observation, reset_info = env.reset(seed=args.seed)
        observation_contract = describe_observation(observation)
        images = find_contract_images(observation)
        image_contract: list[dict[str, Any]] = []
        for index, (name, image) in enumerate(images.items()):
            safe_name = name.replace(".", "_").replace("/", "_") or f"image_{index}"
            image_contract.append(
                {"key": name, **_save_image(image, image_dir / f"{safe_name}.png")}
            )

        action_space_contract = describe_action_space(env.action_space)
        action = policy(env.action_space)
        validate_action(env.action_space, action)
        next_observation, reward, terminated, truncated, info = env.step(action)
        report = {
            "probe": "robocasa_contract",
            "python": sys.version,
            "platform": platform.platform(),
            "environment": {
                "id": args.environment_id,
                "split": args.split,
                "seed": args.seed,
                "gymnasium": gymnasium.__version__,
                "mujoco": mujoco.__version__,
                "robosuite": getattr(robosuite, "__version__", "unknown"),
                "robocasa": getattr(robocasa, "__version__", "unknown"),
            },
            "observation": {
                "contract": observation_contract,
                "state_fields": [
                    record for record in observation_contract if record["key"].startswith("state.")
                ],
                "proprioception_fields": [
                    record for record in observation_contract if record["key"].startswith("state.")
                ],
                "image_count": len(images),
                "images": image_contract,
                "reset_info": _json_info(reset_info),
            },
            "action": {
                "policy": policy.name,
                "parameters": {},
                "space": action_space_contract,
                "value": action_to_json(action),
                "value_shape": action_shape(action),
                "value_dtype": _json_info(getattr(action, "dtype", None)),
                "finite": True,
                "within_space": bool(env.action_space.contains(action)),
            },
            "step": {
                "observation": {
                    "contract": describe_observation(next_observation),
                    "images": [name for name in find_contract_images(next_observation)],
                },
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": _json_info(info),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
