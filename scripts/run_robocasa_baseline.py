#!/usr/bin/env python3
"""Run a deterministic RoboCasa clean baseline without loading a VLA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from vla_recovery_bench.robocasa_adapter import (
    RandomPolicy,
    RoboCasaEnvironment,
    ZeroPolicy,
    action_shape,
    describe_action_space,
)


def _git_commit(path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _observation_contract(observation: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        from collections.abc import Mapping

        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
            return
        shape = getattr(value, "shape", ())
        dtype = getattr(value, "dtype", None)
        records.append(
            {
                "key": prefix,
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "shape": [int(dimension) for dimension in shape],
                "dtype": str(dtype) if dtype is not None else None,
            }
        )

    visit("", observation)
    return records


def _action_hash(action: Any) -> str:
    """Hash the serialized action for auditability without storing large arrays."""
    payload = json.dumps(_jsonable(action), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="target")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("zero", "random"), default="zero")
    parser.add_argument("--output", default="/home/pc/VLA/outputs/robocasa_clean_baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0 or args.horizon <= 0:
        raise ValueError("episodes and horizon must be positive")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("source /home/pc/VLA/env.sh before running the baseline")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")

    import gymnasium
    import mujoco
    import robocasa
    import robosuite

    env = RoboCasaEnvironment(args.environment_id, split=args.split)
    policy_class = ZeroPolicy if args.policy == "zero" else RandomPolicy
    policy = policy_class(env.action_space)
    episodes: list[dict[str, Any]] = []
    try:
        for episode_id in range(args.episodes):
            episode_seed = args.seed + episode_id
            observation = env.reset(episode_seed)
            policy.reset()
            first_contract = _observation_contract(observation)
            total_reward = 0.0
            steps = 0
            success = False
            terminated = False
            truncated = False
            latencies_ms: list[float] = []
            action_shapes: list[Any] = []
            action_hashes: list[str] = []
            for _ in range(args.horizon):
                started = time.perf_counter()
                action = policy.act(observation, "pick and place the object")
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                action_shapes.append(action_shape(action))
                action_hashes.append(_action_hash(action))
                transition = env.step(action)
                observation = transition.observation
                total_reward += transition.reward
                steps += 1
                success = bool(transition.info.get("success", False))
                terminated = transition.terminated
                truncated = transition.truncated
                if success or terminated or truncated:
                    break
            episodes.append(
                {
                    "episode_id": episode_id,
                    "seed": episode_seed,
                    "success": success,
                    "steps": steps,
                    "reward": total_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "policy_inference_latency_ms": {
                        "mean": sum(latencies_ms) / len(latencies_ms),
                        "max": max(latencies_ms),
                        "steps": len(latencies_ms),
                    },
                    "observation_contract": first_contract,
                    "action_shapes": action_shapes,
                    "action_hashes": action_hashes,
                }
            )
    finally:
        env.close()

    successes = sum(bool(episode["success"]) for episode in episodes)
    report = {
        "status": "passed",
        "experiment": "robocasa_clean_baseline",
        "environment": {
            "id": args.environment_id,
            "split": args.split,
            "seed": args.seed,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "python": sys.version,
            "platform": platform.platform(),
            "gymnasium": _package_version(gymnasium),
            "mujoco": _package_version(mujoco),
            "robocasa": _package_version(robocasa),
            "robosuite": _package_version(robosuite),
            "robocasa_commit": _git_commit("/home/pc/VLA/src/robocasa"),
            "robosuite_commit": _git_commit("/home/pc/VLA/src/robosuite"),
        },
        "policy": {
            "name": policy.name,
            "frozen": True,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "scientific_result": False,
        },
        "action_space": describe_action_space(env.action_space),
        "episodes": episodes,
        "metrics": {
            "episodes": len(episodes),
            "successful_episodes": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
            "mean_episode_steps": (
                sum(int(episode["steps"]) for episode in episodes) / len(episodes)
                if episodes
                else 0.0
            ),
            "mean_reward": (
                sum(float(episode["reward"]) for episode in episodes) / len(episodes)
                if episodes
                else 0.0
            ),
        },
    }
    report_path = output_dir / "baseline.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
