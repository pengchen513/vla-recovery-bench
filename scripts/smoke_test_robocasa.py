#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/smoke_test")
    parser.add_argument("--environment-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="target")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("MUJOCO_GL must be set to egl before Python starts")

    import gymnasium
    import mujoco
    import numpy as np
    import robocasa
    import robosuite

    from vla_recovery_bench.robocasa_adapter import find_rgb_observations, save_rgb_image

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gymnasium.make(args.environment_id, split=args.split)
    try:
        observation, info = env.reset(seed=args.seed)
        images = find_rgb_observations(observation)
        if not images:
            raise RuntimeError(f"no HxWx3 RGB observation found; keys={list(observation.keys())}")
        image_key, image = max(images.items(), key=lambda item: np.asarray(item[1]).size)
        stats = save_rgb_image(image, output_dir / "first_frame.png")
        if stats["standard_deviation"] <= 0.0:
            raise RuntimeError("rendered image is blank")
        report = {
            "status": "passed",
            "python": sys.version,
            "platform": platform.platform(),
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "robosuite": getattr(robosuite, "__version__", "unknown"),
            "robocasa": getattr(robocasa, "__version__", "unknown"),
            "gymnasium": getattr(gymnasium, "__version__", "unknown"),
            "environment_id": args.environment_id,
            "split": args.split,
            "seed": args.seed,
            "observation_keys": sorted(observation.keys()),
            "selected_image_key": image_key,
            "image": stats,
            "reset_info_keys": sorted(info.keys()),
        }
        (output_dir / "smoke_test.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

