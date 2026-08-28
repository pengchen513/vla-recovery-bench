#!/usr/bin/env python3
"""Run a no-fault 30-episode RoboCasa baseline with the frozen GR00T policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from vla_recovery_bench.groot_adapter import (
    GrootRoboCasaPolicy,
    TorchZmqPolicyClient,
    flatten_observation,
    task_description,
)
from vla_recovery_bench.robocasa_adapter import (
    RoboCasaEnvironment,
    action_shape,
    describe_action_space,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(manifest: dict[str, Any]) -> dict[str, str]:
    checkpoint_dir = Path(manifest["checkpoint_path"])
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    verified: dict[str, str] = {}
    for record in manifest["checkpoint_files"]:
        path = checkpoint_dir / record["path"]
        expected_size = int(record["size"])
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(
                f"checkpoint size mismatch for {path}: "
                f"expected={expected_size}, got={path.stat().st_size if path.exists() else None}"
            )
        actual = _sha256(path)
        if actual != record["sha256"]:
            raise ValueError(
                f"checkpoint SHA256 mismatch for {path}: "
                f"expected={record['sha256']}, got={actual}"
            )
        verified[record["path"]] = actual
    return verified


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


def _gpu_info() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    result = []
    for line in lines:
        index, name, memory, driver = (part.strip() for part in line.split(",", 3))
        result.append(
            {
                "index": index,
                "name": name,
                "memory_mib": memory,
                "driver": driver,
            }
        )
    return result


def _package_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _observation_shapes(observation: dict[str, Any]) -> dict[str, list[int]]:
    return {
        key: [int(dimension) for dimension in getattr(value, "shape", ())]
        for key, value in flatten_observation(observation).items()
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _start_server(args: argparse.Namespace, output_dir: Path) -> tuple[Any, Any]:
    server_log = (output_dir / "policy_server.log").open("x", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["NO_ALBUMENTATIONS_UPDATE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [
            args.policy_python,
            args.server_script,
            "--checkpoint",
            str(args.checkpoint),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--device",
            "cuda:0",
            "--denoising-steps",
            str(args.denoising_steps),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + args.server_start_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                server_log.flush()
                raise RuntimeError(
                    f"policy server exited with code {process.returncode}; "
                    f"see {output_dir / 'policy_server.log'}"
                )
            client = TorchZmqPolicyClient(port=args.port, timeout_ms=1000)
            try:
                if client.ping():
                    return process, server_log
            finally:
                client.close()
            time.sleep(1.0)
        raise TimeoutError(
            f"policy server did not become ready within {args.server_start_timeout}s"
        )
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        server_log.close()
        raise


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(
        "/home/pc/VLA/checkpoints/groot_atomic_seen_30p/"
        "gr00t_n1-5/target_fraction/atomic_seen_30p"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="target", choices=("target",))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=750)
    parser.add_argument(
        "--manifest",
        default=str(root / "configs/policies/groot_n1_5_robocasa_atomic_seen_30p.json"),
    )
    parser.add_argument("--checkpoint", type=Path, default=checkpoint)
    parser.add_argument("--policy-python", default="/home/pc/VLA/envs/groot/bin/python")
    parser.add_argument("--server-script", default=str(root / "scripts/serve_groot_policy.py"))
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--server-start-timeout", type=int, default=900)
    parser.add_argument(
        "--output", default="/home/pc/VLA/outputs/groot_atomic_seen_30p_clean_baseline"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes != 30 or args.seed_start != 0:
        raise ValueError("scientific clean baseline requires exactly seeds 0 through 29")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("source /home/pc/VLA/env.sh before running the baseline")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")

    manifest = _load_json(args.manifest)
    if Path(manifest["checkpoint_path"]) != args.checkpoint:
        raise ValueError("manifest checkpoint_path does not match --checkpoint")
    checkpoint_hashes = verify_checkpoint(manifest)

    import gymnasium
    import mujoco
    import robocasa
    import robosuite
    import torch

    process = None
    server_log = None
    client = None
    env = None
    episodes_path = output_dir / "episodes.jsonl"
    episode_records: list[dict[str, Any]] = []
    inference_latencies: list[float] = []
    parameter_state_before: dict[str, Any] | None = None
    parameter_state_after: dict[str, Any] | None = None
    try:
        process, server_log = _start_server(args, output_dir)
        client = TorchZmqPolicyClient(port=args.port, timeout_ms=300000)
        parameter_state_before = client.call("get_policy_state")
        if parameter_state_before["model_training"]:
            raise RuntimeError("GR00T model is not in eval mode")
        if not parameter_state_before["all_parameters_frozen"]:
            raise RuntimeError("GR00T model contains trainable parameters")

        env = RoboCasaEnvironment(args.environment_id, split=args.split)
        policy = GrootRoboCasaPolicy(env.action_space, client)
        action_space = describe_action_space(env.action_space)
        first_observation_shapes: dict[str, list[int]] | None = None
        with episodes_path.open("x", encoding="utf-8") as stream:
            for episode_id, seed in enumerate(range(args.seed_start, args.seed_start + 30)):
                observation = env.reset(seed)
                policy.reset()
                seed_state = client.call("set_seed", {"seed": seed})
                if seed_state != {"seed": seed}:
                    raise RuntimeError(f"policy server did not accept seed {seed}")
                if first_observation_shapes is None:
                    first_observation_shapes = _observation_shapes(observation)
                total_reward = 0.0
                success = False
                terminated = False
                truncated = False
                action_shapes = None
                episode_inference_latencies: list[float] = []
                saturation_chunks: list[dict[str, Any]] = []
                started = time.perf_counter()
                steps = 0
                for _ in range(args.horizon):
                    prompt = task_description(observation)
                    action = policy.act(observation, prompt)
                    action_shapes = action_shape(action)
                    if policy.last_inference_latency_ms is not None:
                        latency = policy.last_inference_latency_ms
                        inference_latencies.append(latency)
                        episode_inference_latencies.append(latency)
                        if policy.last_chunk_saturation is None:
                            raise RuntimeError("missing action saturation audit")
                        saturation_chunks.append(policy.last_chunk_saturation)
                    transition = env.step(action)
                    observation = transition.observation
                    total_reward += transition.reward
                    steps += 1
                    success = bool(transition.info.get("success", False))
                    terminated = transition.terminated
                    truncated = transition.truncated
                    if success or terminated or truncated:
                        break
                record = {
                    "episode_id": episode_id,
                    "seed": seed,
                    "success": success,
                    "steps": steps,
                    "reward": total_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "wall_time_seconds": time.perf_counter() - started,
                    "policy_inferences": len(episode_inference_latencies),
                    "policy_inference_latency_ms": {
                        "mean": statistics.fmean(episode_inference_latencies),
                        "maximum": max(episode_inference_latencies),
                    },
                    "action_saturation": {
                        "chunks": len(saturation_chunks),
                        "saturated_values": policy.episode_saturated_values,
                        "chunk_audits": saturation_chunks,
                    },
                    "observation_shapes": first_observation_shapes,
                    "action_shapes": action_shapes,
                }
                episode_records.append(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                print(json.dumps(record, sort_keys=True), flush=True)

        parameter_state_after = client.call("get_policy_state")
        if (
            parameter_state_before["current_parameter_sha256"]
            != parameter_state_after["current_parameter_sha256"]
        ):
            raise RuntimeError("GR00T model parameter hash changed during clean evaluation")

        successful = sum(bool(record["success"]) for record in episode_records)
        metrics = {
            "status": "completed",
            "experiment": "groot_n1_5_robocasa_clean_baseline",
            "faults_enabled": False,
            "environment": {
                "id": args.environment_id,
                "split": args.split,
                "seeds": list(range(30)),
                "episodes": 30,
                "horizon": args.horizon,
                "observation_shapes": first_observation_shapes,
                "action_space": action_space,
                "robocasa_commit": _git_commit("/home/pc/VLA/src/robocasa"),
                "robosuite_commit": _git_commit("/home/pc/VLA/src/robosuite"),
            },
            "policy": {
                "name": GrootRoboCasaPolicy.name,
                "manifest": str(Path(args.manifest).resolve()),
                "checkpoint": str(args.checkpoint),
                "checkpoint_files_sha256": checkpoint_hashes,
                "groot_commit": _git_commit("/home/pc/VLA/src/Isaac-GR00T"),
                "denoising_steps": args.denoising_steps,
                "action_chunk_length": 16,
                "frozen": True,
                "parameter_state_before": parameter_state_before,
                "parameter_state_after": parameter_state_after,
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch_client": _package_version(torch),
                "gymnasium": _package_version(gymnasium),
                "mujoco": _package_version(mujoco),
                "robocasa": _package_version(robocasa),
                "robosuite": _package_version(robosuite),
            },
            "gpu": _gpu_info(),
            "metrics": {
                "episodes": len(episode_records),
                "successful_episodes": successful,
                "success_rate": successful / len(episode_records),
                "mean_episode_steps": statistics.fmean(
                    float(record["steps"]) for record in episode_records
                ),
                "mean_reward": statistics.fmean(
                    float(record["reward"]) for record in episode_records
                ),
                "policy_inference_latency_ms": {
                    "count": len(inference_latencies),
                    "mean": statistics.fmean(inference_latencies),
                    "p50": _percentile(inference_latencies, 50),
                    "p95": _percentile(inference_latencies, 95),
                    "maximum": max(inference_latencies),
                },
            },
            "official_reference": manifest["published_clean_baseline"],
            "outputs": {
                "episodes_jsonl": str(episodes_path),
                "policy_server_log": str(output_dir / "policy_server.log"),
            },
        }
        metrics_path = output_dir / "metrics.json"
        metrics["outputs"]["metrics_json"] = str(metrics_path)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    finally:
        if env is not None:
            env.close()
        if client is not None:
            try:
                client.call("kill")
            except Exception:
                pass
            client.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
