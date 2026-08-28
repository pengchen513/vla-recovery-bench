#!/usr/bin/env python3
"""Record a short, model-free RoboCasa rollout as an MP4 video.

This is a visual/interface smoke run only.  It does not load a checkpoint or
inject a fault.  ``random`` is useful for seeing the robot move; neither
policy is a scientific baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/home/pc/VLA/outputs/robocasa_clean_30s.mp4",
        help="MP4 path; an existing file is never overwritten unless --overwrite is used",
    )
    parser.add_argument(
        "--environment-id", default="robocasa/PickPlaceCounterToCabinet"
    )
    parser.add_argument("--split", default="target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--policy",
        choices=("random", "zero"),
        default="random",
        help="model-free action source; random visibly moves the robot",
    )
    parser.add_argument(
        "--camera",
        default="video.robot0_agentview_left",
        help="camera observation key to record",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _as_rgb_uint8(value: Any):
    import numpy as np

    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected HxWx3 or HxWx4 camera frame, got {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32, copy=False)
        if float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    # MuJoCo camera observations are vertically flipped in this configuration.
    return np.ascontiguousarray(np.flipud(array))


def _start_ffmpeg(
    output: Path, width: int, height: int, fps: int, overwrite: bool
) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required; install it or provide it on PATH")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_video(
    *,
    output: Path,
    environment_id: str,
    split: str,
    seed: int,
    seconds: float,
    fps: int,
    policy_name: str,
    camera_key: str,
    overwrite: bool,
) -> dict[str, Any]:
    from vla_recovery_bench.robocasa_adapter import (
        RandomPolicy,
        RoboCasaEnvironment,
        ZeroPolicy,
        find_rgb_observations,
    )

    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("source /home/pc/VLA/env.sh before running this script")
    if seconds <= 0 or fps <= 0:
        raise ValueError("seconds and fps must be positive")
    frame_count = int(round(seconds * fps))
    if frame_count <= 0:
        raise ValueError("seconds * fps must produce at least one frame")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing video: {output}; "
            "choose another --output or use --overwrite"
        )
    report_path = output.with_suffix(".json")
    if report_path.exists() and report_path.stat().st_size > 0 and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing video report: {report_path}; "
            "choose another --output or use --overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    environment = RoboCasaEnvironment(environment_id, split=split)
    policy_class = RandomPolicy if policy_name == "random" else ZeroPolicy
    policy = policy_class(environment.action_space)
    ffmpeg_process: subprocess.Popen[bytes] | None = None
    observation = environment.reset(seed)
    episode_count = 1
    selected_camera = camera_key
    frames_written = 0
    frame_shape: list[int] | None = None
    try:
        for _frame_id in range(frame_count):
            images = find_rgb_observations(observation)
            if selected_camera not in images:
                available = ", ".join(sorted(images))
                raise KeyError(f"camera {selected_camera!r} not found; available: {available}")
            frame = _as_rgb_uint8(images[selected_camera])
            if frame_shape is None:
                frame_shape = [int(dimension) for dimension in frame.shape]
                ffmpeg_process = _start_ffmpeg(
                    output,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    fps=fps,
                    overwrite=overwrite,
                )
            elif list(frame.shape) != frame_shape:
                raise ValueError(f"camera frame shape changed: {frame.shape} != {frame_shape}")
            assert ffmpeg_process is not None and ffmpeg_process.stdin is not None
            ffmpeg_process.stdin.write(frame.tobytes())
            frames_written += 1

            action = policy.act(observation, "pick and place the object")
            transition = environment.step(action)
            observation = transition.observation
            if transition.terminated or transition.truncated:
                episode_count += 1
                observation = environment.reset(seed + episode_count - 1)
                policy.reset()
    except (BrokenPipeError, OSError):
        if ffmpeg_process is not None:
            ffmpeg_process.kill()
        raise
    finally:
        environment.close()

    if ffmpeg_process is None or ffmpeg_process.stdin is None:
        raise RuntimeError("no video frames were produced")
    ffmpeg_process.stdin.close()
    stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="replace")
    return_code = ffmpeg_process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr.strip()}")
    if frames_written != frame_count:
        raise RuntimeError(f"wrote {frames_written} frames, expected {frame_count}")
    return {
        "status": "passed",
        "video": str(output),
        "environment_id": environment_id,
        "split": split,
        "seed": seed,
        "policy": policy_name,
        "faults": False,
        "model_checkpoint": None,
        "seconds": seconds,
        "fps": fps,
        "frames": frames_written,
        "episodes": episode_count,
        "camera": camera_key,
        "frame_shape": frame_shape,
    }


def main() -> int:
    args = parse_args()
    report = _write_video(
        output=Path(args.output),
        environment_id=args.environment_id,
        split=args.split,
        seed=args.seed,
        seconds=args.seconds,
        fps=args.fps,
        policy_name=args.policy,
        camera_key=args.camera,
        overwrite=args.overwrite,
    )
    report_path = Path(args.output).with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
