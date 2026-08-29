#!/usr/bin/env python3
"""Collect separated Phase-1 monitor inputs and offline labels with frozen GR00T.

This command does not train a policy, run a recovery action, or expose the
fault schedule to an online monitor.  Final-test seeds are intentionally
rejected; they require a separately locked monitor-evaluation command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vla_recovery_bench.artifacts import ensure_empty_output_dir, write_json_once
from vla_recovery_bench.faults import FaultSchedule
from vla_recovery_bench.groot_adapter import (
    GrootRoboCasaPolicy,
    TorchZmqPolicyClient,
    task_description,
)
from vla_recovery_bench.monitor import FEATURE_NAMES, FEATURE_VERSION, context_to_feature
from vla_recovery_bench.monitor_dataset import MonitorDatasetWriter, validate_monitor_dataset
from vla_recovery_bench.monitor_gate import build_shard_integrity_manifest
from vla_recovery_bench.monitor_protocol import (
    monitor_episode_plan,
    validate_monitor_relock_protocol,
)
from vla_recovery_bench.recording import to_jsonable
from vla_recovery_bench.robocasa_adapter import RoboCasaEnvironment, describe_action_space
from vla_recovery_bench.types import ActionChunkMetadata, AuditRecord, FaultPhase, MonitorContext

try:
    from scripts.run_identifiability_pilot import (
        DEFAULT_CHECKPOINT,
        DEFAULT_MANIFEST,
        ROOT,
        _git_commit,
        _git_dirty,
        _gpu_info,
        _load_json,
        _package_versions,
        _sha256,
        _start_server,
        _verify_checkpoint,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_identifiability_pilot import (  # type: ignore[no-redef]
        DEFAULT_CHECKPOINT,
        DEFAULT_MANIFEST,
        ROOT,
        _git_commit,
        _git_dirty,
        _gpu_info,
        _load_json,
        _package_versions,
        _sha256,
        _start_server,
        _verify_checkpoint,
    )

DEFAULT_PROTOCOL = ROOT / "configs/monitor_relock_v1_2.json"


def _record(stream: Any, event_type: str, **payload: Any) -> None:
    stream.write(
        json.dumps(to_jsonable({"event_type": event_type, **payload}), sort_keys=True) + "\n"
    )
    stream.flush()


def _episode_token(protocol_sha256: str, episode_id: str) -> str:
    return hashlib.sha256(
        f"monitor-input-channel-v1|{protocol_sha256}|{episode_id}".encode()
    ).hexdigest()[:32]


def _is_exposed(
    *, condition: str, control_step: int, observation_step: int, factor_row: Mapping[str, Any]
) -> bool:
    if condition == "clean":
        return False
    onset = int(factor_row["onset_step"])
    end = onset + int(factor_row["duration_steps"])
    reference = observation_step if condition == "observation_fault" else control_step
    return onset <= reference < end


def _run_episode(
    *,
    episode_index: int,
    item: Mapping[str, Any],
    environment: RoboCasaEnvironment,
    policy: GrootRoboCasaPolicy,
    client: TorchZmqPolicyClient,
    horizon: int,
    protocol_sha256: str,
    writer: MonitorDatasetWriter,
    index_stream: Any,
    episode_stream: Any,
    audit_stream: Any,
) -> dict[str, Any]:
    seed = int(item["seed"])
    condition = str(item["condition"])
    mechanism = str(item["mechanism"])
    token = _episode_token(protocol_sha256, str(item["episode_id"]))
    schedule = FaultSchedule(item["faults"])
    schedule.reset()
    response = client.call("set_seed", {"seed": seed})
    if response != {"seed": seed}:
        raise RuntimeError(f"policy server did not accept seed {seed}")
    policy.reset()
    observation = environment.reset(seed)
    features: list[np.ndarray] = []
    control_steps: list[int] = []
    observation_steps: list[int] = []
    exposures: list[bool] = []
    policy_latencies: list[float] = []
    fault_events: list[dict[str, Any]] = []
    reward = 0.0
    success = False
    terminated = False
    truncated = False
    termination_reason = "horizon"
    chunk_length = int(policy.action_chunk_length)
    chunk_id = 0
    chunk_position = 0
    started = time.perf_counter()
    prompt = task_description(observation)

    for step in range(horizon):
        for fault in schedule.due(step, FaultPhase.BEFORE_ACTION):
            application = environment.inject_fault(fault)
            if not application.applied:
                raise RuntimeError(f"declared monitor fault was not applied: {application}")
            event = {
                "episode_index": episode_index,
                "episode_id": item["episode_id"],
                "episode_token": token,
                "condition": condition,
                "seed": seed,
                "control_step": step,
                "phase": FaultPhase.BEFORE_ACTION,
                "fault": fault,
                "application": application,
            }
            fault_events.append(event)
            _record(audit_stream, "fault_injection", **event)

        prompt = task_description(observation)
        action = policy.act(observation, prompt)
        if policy.last_inference_latency_ms is not None:
            policy_latencies.append(float(policy.last_inference_latency_ms))
        transition = environment.step(action)
        reward += float(transition.reward)

        after_fault_applied = False
        for fault in schedule.due(step, FaultPhase.AFTER_STEP):
            application = environment.inject_fault(fault)
            if not application.applied:
                raise RuntimeError(f"declared monitor fault was not applied: {application}")
            after_fault_applied = True
            event = {
                "episode_index": episode_index,
                "episode_id": item["episode_id"],
                "episode_token": token,
                "condition": condition,
                "seed": seed,
                "control_step": step,
                "phase": FaultPhase.AFTER_STEP,
                "fault": fault,
                "application": application,
            }
            fault_events.append(event)
            _record(audit_stream, "fault_injection", **event)
        if after_fault_applied:
            transition = replace(
                transition,
                observation=environment.apply_pending_observation_fault(
                    transition.observation
                ),
            )

        requested_chunk = policy.requested_action_chunk
        if len(requested_chunk) != chunk_length:
            raise RuntimeError(
                "policy did not expose the complete requested action chunk: "
                f"expected={chunk_length}, got={len(requested_chunk)}"
            )
        chunk = ActionChunkMetadata(
            chunk_id=chunk_id,
            position_in_chunk=chunk_position,
            chunk_length=chunk_length,
            remaining_horizon=max(horizon - step - 1, 0),
            policy_inference_latency_ms=policy.last_inference_latency_ms,
        )
        context = MonitorContext(
            episode_id=0,
            step=step + 1,
            instruction=prompt,
            previous_observation=observation,
            observation=transition.observation,
            action=action,
            action_chunk=requested_chunk,
            chunk=chunk,
        )
        features.append(context_to_feature(context))
        control_steps.append(step)
        observation_steps.append(step + 1)
        exposures.append(
            _is_exposed(
                condition=condition,
                control_step=step,
                observation_step=step + 1,
                factor_row=item["factor_row"],
            )
        )
        _record(
            audit_stream,
            "audit_transition",
            episode_index=episode_index,
            episode_id=item["episode_id"],
            episode_token=token,
            pair_id=item["pair_id"],
            condition=condition,
            seed=seed,
            control_step=step,
            audit=AuditRecord(
                episode_id=episode_index,
                step=step,
                requested_action=action,
                executed_action=transition.executed_action,
                reward=transition.reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                info=transition.info,
                success=bool(transition.info.get("success", False)),
            ),
        )

        observation = transition.observation
        success = bool(transition.info.get("success", False))
        terminated = bool(transition.terminated)
        truncated = bool(transition.truncated)
        if chunk_position + 1 >= chunk_length:
            chunk_id += 1
            chunk_position = 0
        else:
            chunk_position += 1
        if success:
            termination_reason = "success"
            break
        if terminated or truncated:
            termination_reason = "environment_done"
            break

    steps = len(features)
    feature_array = np.stack(features, axis=0)
    exposure_array = np.asarray(exposures, dtype=np.bool_)
    label = {
        "episode_id": str(item["episode_id"]),
        "pair_id": str(item["pair_id"]),
        "partition": str(item["partition"]),
        "seed": seed,
        "condition": condition,
        "mechanism": mechanism,
        "factor_row": dict(item["factor_row"]),
        "fault_schedule": to_jsonable(item["faults"]),
        "success": success,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": termination_reason,
    }
    writer.write_episode(
        token=token,
        features=feature_array,
        control_steps=np.asarray(control_steps),
        observation_steps=np.asarray(observation_steps),
        instruction=prompt,
        label=label,
        exposure=exposure_array,
    )
    index = {
        "episode_token": token,
        "episode_id": item["episode_id"],
        "pair_id": item["pair_id"],
        "partition": item["partition"],
        "seed": seed,
        "feature_rows": steps,
        "input_group": f"/episodes/{token}",
        "label_group": f"/episodes/{token}",
    }
    _record(index_stream, "dataset_episode", **index)
    summary = {
        **index,
        "condition": condition,
        "mechanism": mechanism,
        "configured_fault_count": len(item["faults"]),
        "applied_fault_count": len(fault_events),
        "exposed_rows": int(exposure_array.sum()),
        "not_exposed": bool(condition != "clean" and not exposure_array.any()),
        "success": success,
        "steps": steps,
        "reward": reward,
        "termination_reason": termination_reason,
        "wall_time_seconds": time.perf_counter() - started,
        "policy_inference_latency_ms": {
            "mean": statistics.fmean(policy_latencies) if policy_latencies else None,
            "p95": float(np.percentile(policy_latencies, 95)) if policy_latencies else None,
            "maximum": max(policy_latencies) if policy_latencies else None,
        },
        "action_saturated_values": int(policy.episode_saturated_values),
    }
    _record(episode_stream, "episode", **summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--partition", choices=("train", "calibration", "validation"), required=True
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--collection-role",
        choices=("debug", "formal_shard", "full_partition"),
        default="debug",
    )
    parser.add_argument("--max-scene-seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--scene-seed", type=int, default=None)
    parser.add_argument("--environment-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="target", choices=("target",))
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--policy-python", default="/home/pc/VLA/envs/groot/bin/python")
    parser.add_argument("--server-script", default=str(ROOT / "scripts/serve_groot_policy.py"))
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--server-start-timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("source /home/pc/VLA/env.sh before collecting monitor data")
    protocol = _load_json(args.protocol)
    manifest = _load_json(args.manifest)
    if protocol.get("relock_version") is not None:
        reference = Path(str(protocol.get("parent_monitor_protocol", "")))
        candidates = [
            args.protocol.parent / reference,
            args.protocol.resolve().parents[1] / reference,
            reference,
        ]
        parent_path = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()), None
        )
        if parent_path is None:
            raise ValueError(f"monitor relock parent protocol is missing: {reference}")
        parent = _load_json(parent_path)
        parent_hash = _sha256(parent_path)
        relock_errors = validate_monitor_relock_protocol(
            protocol, parent_config=parent, parent_sha256=parent_hash
        )
        if relock_errors:
            raise ValueError(f"invalid monitor relock protocol: {relock_errors}")
    configured_horizon = int(protocol["environment"]["horizon"])
    horizon = configured_horizon if args.horizon is None else int(args.horizon)
    if horizon <= 0 or horizon > configured_horizon:
        raise ValueError(f"horizon must be in [1, {configured_horizon}]")
    split_seeds = list(protocol["splits"][f"{args.partition}_scene_seeds"])
    if args.seed_offset < 0 or args.seed_offset >= len(split_seeds):
        raise ValueError(
            f"--seed-offset must be in [0, {len(split_seeds) - 1}] for {args.partition}"
        )
    if args.max_scene_seeds is not None and args.scene_seed is not None:
        raise ValueError("use only one of --max-scene-seeds and --scene-seed")
    if args.scene_seed is not None:
        if args.scene_seed not in split_seeds:
            raise ValueError(
                f"--scene-seed {args.scene_seed} is outside partition {args.partition}"
            )
        selected_seeds = [args.scene_seed]
    elif args.max_scene_seeds is not None:
        if args.max_scene_seeds <= 0:
            raise ValueError("--max-scene-seeds must be positive")
        selected_seeds = split_seeds[
            args.seed_offset : args.seed_offset + args.max_scene_seeds
        ]
        if len(selected_seeds) != args.max_scene_seeds:
            raise ValueError(
                "requested shard overruns the declared partition: "
                f"offset={args.seed_offset}, requested={args.max_scene_seeds}, "
                f"available={len(selected_seeds)}"
            )
    else:
        if args.seed_offset:
            raise ValueError("--seed-offset requires --max-scene-seeds")
        selected_seeds = split_seeds
    plan = monitor_episode_plan(protocol, args.partition, seeds=selected_seeds)
    expected_count = len(selected_seeds) * 3
    if len(plan) != expected_count:
        raise RuntimeError(
            f"monitor collection plan has {len(plan)} episodes; expected {expected_count}"
        )
    protocol_sha256 = _sha256(args.protocol)
    if protocol["policy"]["name"] != manifest["policy_name"]:
        raise ValueError("monitor protocol and policy manifest names disagree")
    if protocol["policy"]["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("monitor protocol and policy manifest checkpoint hashes disagree")
    partition_complete = selected_seeds == split_seeds
    debug = args.collection_role == "debug"
    if debug and horizon == configured_horizon:
        raise ValueError("debug collection must use a reduced --horizon")
    if not debug and horizon != configured_horizon:
        raise ValueError("formal collection must use the frozen full horizon")
    if args.collection_role == "formal_shard" and partition_complete:
        raise ValueError("formal_shard must select a strict subset of the partition")
    if args.collection_role == "full_partition" and not partition_complete:
        raise ValueError("full_partition must select every declared partition seed")
    if not debug and _git_dirty(ROOT) is not False:
        raise RuntimeError(
            "formal collection requires a clean repository snapshot; "
            "commit the frozen protocol and implementation before collecting"
        )
    checkpoint_hashes = _verify_checkpoint(manifest, args.checkpoint)
    output_dir = ensure_empty_output_dir(args.output)
    paths = {
        "run_manifest": output_dir / "run_manifest.json",
        "dataset_index": output_dir / "dataset_index.jsonl",
        "monitor_inputs": output_dir / "monitor_inputs.h5",
        "offline_labels": output_dir / "offline_labels.h5",
        "episodes": output_dir / "episodes.jsonl",
        "audit_stream": output_dir / "audit_stream.jsonl",
        "metrics": output_dir / "metrics.json",
        "software_versions": output_dir / "software_versions.json",
        "policy_state_before": output_dir / "policy_state_before.json",
        "policy_state_after": output_dir / "policy_state_after.json",
        "artifact_validation": output_dir / "artifact_validation.json",
        "shard_integrity": output_dir / "shard_integrity.json",
    }
    write_json_once(
        paths["software_versions"],
        {
            "packages": _package_versions(),
            "gpu": _gpu_info(),
            "repository_commit": _git_commit(ROOT),
            "repository_dirty": _git_dirty(ROOT),
            "robocasa_commit": _git_commit("/home/pc/VLA/src/robocasa"),
            "robosuite_commit": _git_commit("/home/pc/VLA/src/robosuite"),
            "groot_commit": _git_commit("/home/pc/VLA/src/Isaac-GR00T"),
        },
    )

    process = None
    server_log = None
    client = None
    environment = None
    results: list[dict[str, Any]] = []
    try:
        process, server_log = _start_server(args, output_dir)
        client = TorchZmqPolicyClient(port=args.port, timeout_ms=300000)
        policy_state_before = client.call("get_policy_state")
        if (
            policy_state_before["model_training"]
            or not policy_state_before["all_parameters_frozen"]
        ):
            raise RuntimeError("policy server did not report eval mode with frozen parameters")
        write_json_once(paths["policy_state_before"], policy_state_before)
        environment = RoboCasaEnvironment(args.environment_id, split=args.split)
        policy = GrootRoboCasaPolicy(environment.action_space, client)
        with (
            MonitorDatasetWriter(
                output_dir,
                partition=args.partition,
                protocol_sha256=protocol_sha256,
            ) as writer,
            paths["dataset_index"].open("x", encoding="utf-8") as index_stream,
            paths["episodes"].open("x", encoding="utf-8") as episode_stream,
            paths["audit_stream"].open("x", encoding="utf-8") as audit_stream,
        ):
            for index, item in enumerate(plan):
                result = _run_episode(
                    episode_index=index,
                    item=item,
                    environment=environment,
                    policy=policy,
                    client=client,
                    horizon=horizon,
                    protocol_sha256=protocol_sha256,
                    writer=writer,
                    index_stream=index_stream,
                    episode_stream=episode_stream,
                    audit_stream=audit_stream,
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "episode_id",
                                "seed",
                                "condition",
                                "success",
                                "steps",
                                "exposed_rows",
                            )
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        policy_state_after = client.call("get_policy_state")
        if (
            policy_state_before["current_parameter_sha256"]
            != policy_state_after["current_parameter_sha256"]
        ):
            raise RuntimeError("frozen GR00T parameter hash changed during monitor data collection")
        write_json_once(paths["policy_state_after"], policy_state_after)
        metrics = {
            "status": "completed",
            "scientific_result": False,
            "debug": debug,
            "collection_role": args.collection_role,
            "partition_complete": partition_complete,
            "protocol_version": protocol["protocol_version"],
            "monitor_protocol_version": protocol["monitor_protocol_version"],
            "relock_version": protocol.get("relock_version"),
            "parent_monitor_protocol": protocol.get("parent_monitor_protocol"),
            "parent_monitor_protocol_sha256": protocol.get(
                "parent_monitor_protocol_sha256"
            ),
            "partition": args.partition,
            "seeds": selected_seeds,
            "episode_count": len(results),
            "rows": sum(int(result["feature_rows"]) for result in results),
            "conditions": {
                condition: sum(result["condition"] == condition for result in results)
                for condition in ("clean", "actuator_fault", "observation_fault")
            },
            "exposed_fault_episodes": sum(
                result["condition"] != "clean" and result["exposed_rows"] > 0
                for result in results
            ),
            "not_exposed_fault_episodes": sum(bool(result["not_exposed"]) for result in results),
            "feature_version": FEATURE_VERSION,
            "feature_count": len(FEATURE_NAMES),
            "environment": {
                "id": args.environment_id,
                "split": args.split,
                "horizon": horizon,
                "action_space": describe_action_space(environment.action_space),
            },
            "policy": {
                "name": manifest["policy_name"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "checkpoint_files_sha256": checkpoint_hashes,
                "parameter_sha256_before": policy_state_before["current_parameter_sha256"],
                "parameter_sha256_after": policy_state_after["current_parameter_sha256"],
                "frozen": True,
            },
            "outputs": {key: str(path) for key, path in paths.items()},
        }
        write_json_once(paths["metrics"], metrics)
        write_json_once(
            paths["run_manifest"],
            {
                "protocol_version": protocol["protocol_version"],
                "monitor_protocol_version": protocol["monitor_protocol_version"],
                "relock_version": protocol.get("relock_version"),
                "parent_monitor_protocol": protocol.get("parent_monitor_protocol"),
                "parent_monitor_protocol_sha256": protocol.get(
                    "parent_monitor_protocol_sha256"
                ),
                "status": "completed",
                "scientific_result": False,
                "debug": debug,
                "collection_role": args.collection_role,
                "partition_complete": partition_complete,
                "partition": args.partition,
                "environment": metrics["environment"],
                "policy": metrics["policy"],
                "monitor_inputs": protocol["information_boundary"],
                "storage": protocol["storage"],
                "seeds": selected_seeds,
                "episode_plan": plan,
                "config": {"path": str(args.protocol.resolve()), "sha256": protocol_sha256},
                "policy_manifest": {
                    "path": str(args.manifest.resolve()),
                    "sha256": _sha256(args.manifest),
                },
                "command": [sys.executable, *sys.argv],
                "artifacts": metrics["outputs"],
            },
        )
        errors = validate_monitor_dataset(
            output_dir,
            expected_partition=args.partition,
            expected_episode_count=expected_count,
        )
        write_json_once(
            paths["artifact_validation"],
            {"status": "passed" if not errors else "failed", "errors": errors},
        )
        if errors:
            raise RuntimeError(f"monitor dataset artifact validation failed: {errors}")
        write_json_once(
            paths["shard_integrity"],
            build_shard_integrity_manifest(
                output_dir,
                partition=args.partition,
                collection_role=args.collection_role,
                seeds=selected_seeds,
                protocol_sha256=protocol_sha256,
                policy_manifest_sha256=_sha256(args.manifest),
            ),
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    finally:
        if environment is not None:
            environment.close()
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
            except TimeoutError:
                process.kill()
                process.wait(timeout=30)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
