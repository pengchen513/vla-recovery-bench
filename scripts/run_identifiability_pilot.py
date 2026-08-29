#!/usr/bin/env python3
"""Run the pre-registered v1.4 RoboCasa identifiability pilot.

The pilot is an evaluation-only gate.  It uses the already audited frozen GR00T
checkpoint, never exposes the fault schedule or simulator outcomes to the
monitor, and writes monitor and privileged audit streams to different files.
"""

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
from collections import Counter, defaultdict
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
    flatten_observation,
    task_description,
)
from vla_recovery_bench.pilot import pilot_episode_plan, validate_pilot_artifacts
from vla_recovery_bench.robocasa_adapter import RoboCasaEnvironment, describe_action_space
from vla_recovery_bench.types import (
    ActionChunkMetadata,
    AuditRecord,
    FaultPhase,
    MonitorContext,
    MonitorDecision,
    RecoveryAction,
    RecoveryContext,
    RecoveryDecision,
)

DEFAULT_CHECKPOINT = Path(
    "/home/pc/VLA/checkpoints/groot_atomic_seen_30p/gr00t_n1-5/target_fraction/atomic_seen_30p"
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/identifiability_pilot_v1_4.json"
DEFAULT_MANIFEST = ROOT / "configs/policies/groot_n1_5_robocasa_atomic_seen_30p.json"


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


def _verify_checkpoint(manifest: Mapping[str, Any], checkpoint: Path) -> dict[str, str]:
    if Path(str(manifest["checkpoint_path"])) != checkpoint:
        raise ValueError("policy manifest checkpoint_path does not match --checkpoint")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    verified: dict[str, str] = {}
    for record in manifest["checkpoint_files"]:
        path = checkpoint / str(record["path"])
        expected_size = int(record["size"])
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"checkpoint size mismatch: {path}")
        actual = _sha256(path)
        if actual != str(record["sha256"]):
            raise ValueError(f"checkpoint SHA256 mismatch: {path}")
        verified[str(record["path"])] = actual
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


def _git_dirty(path: str | Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


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
        result.append({"index": index, "name": name, "memory_mib": memory, "driver": driver})
    return result


def _package_versions() -> dict[str, str]:
    import importlib.metadata

    packages = (
        "torch",
        "gymnasium",
        "mujoco",
        "robocasa",
        "robosuite",
        "numpy",
        "scipy",
    )
    result: dict[str, str] = {"python": sys.version}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    result["platform"] = platform.platform()
    return result


def _start_server(args: argparse.Namespace, output_dir: Path) -> tuple[subprocess.Popen[Any], Any]:
    log_path = output_dir / "policy_server.log"
    log_stream = log_path.open("x", encoding="utf-8")
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
        cwd=ROOT,
        env=environment,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + args.server_start_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"policy server exited with code {process.returncode}; see {log_path}"
                )
            client = TorchZmqPolicyClient(port=args.port, timeout_ms=1000)
            try:
                if client.ping():
                    return process, log_stream
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
        log_stream.close()
        raise


def _flatten_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
        return np.empty(0, dtype=np.float32)
    return array.astype(np.float32, copy=False).reshape(-1)


def _image_features(current: Any, previous: Any | None) -> dict[str, float]:
    image = np.asarray(current)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        return {}
    rgb = image[..., :3].astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        rgb /= 255.0
    features = {
        "mean": float(rgb.mean()),
        "standard_deviation": float(rgb.std()),
        "zero_fraction": float(np.mean(rgb == 0.0)),
    }
    if previous is not None:
        old = np.asarray(previous)[..., :3].astype(np.float32)
        if np.issubdtype(np.asarray(previous).dtype, np.integer):
            old /= 255.0
        if old.shape == rgb.shape:
            features["temporal_abs_difference"] = float(np.mean(np.abs(rgb - old)))
    return features


def _visit_images(observation: Mapping[str, Any]) -> dict[str, Any]:
    images: dict[str, Any] = {}
    for key, value in flatten_observation(observation).items():
        shape = getattr(value, "shape", ())
        if len(shape) == 3 and shape[-1] in (3, 4):
            images[key] = value
    return images


def _monitor_features(context: MonitorContext) -> dict[str, float]:
    current = flatten_observation(context.observation)
    previous = flatten_observation(context.previous_observation)
    current_images = _visit_images(context.observation)
    previous_images = _visit_images(context.previous_observation)
    image_stats = [
        _image_features(value, previous_images.get(key)) for key, value in current_images.items()
    ]
    image_stats = [stats for stats in image_stats if stats]
    state_values = np.concatenate(
        [_flatten_numeric(value) for key, value in current.items() if key.startswith("state.")]
    )
    previous_state_values = np.concatenate(
        [_flatten_numeric(value) for key, value in previous.items() if key.startswith("state.")]
    )
    action_values = (
        np.concatenate([_flatten_numeric(value) for value in context.action.values()])
        if isinstance(context.action, Mapping)
        else _flatten_numeric(context.action)
    )
    state_delta = (
        float(np.linalg.norm(state_values - previous_state_values))
        if state_values.shape == previous_state_values.shape
        else 0.0
    )
    action_norm = float(np.linalg.norm(action_values)) if action_values.size else 0.0
    temporal_difference = (
        float(statistics.fmean(stats.get("temporal_abs_difference", 0.0) for stats in image_stats))
        if image_stats
        else 0.0
    )
    std_mean = (
        float(statistics.fmean(stats.get("standard_deviation", 0.0) for stats in image_stats))
        if image_stats
        else 0.0
    )
    zero_fraction = (
        float(statistics.fmean(stats.get("zero_fraction", 0.0) for stats in image_stats))
        if image_stats
        else 0.0
    )
    temporal_values = [stats.get("temporal_abs_difference", 0.0) for stats in image_stats]
    temporal_range = float(max(temporal_values) - min(temporal_values)) if temporal_values else 0.0
    # These are fixed, transparent evidence scores for the evaluation-only pilot.
    observation_score = min(
        1.0,
        max(zero_fraction * 2.0, temporal_difference * 8.0, temporal_range * 10.0),
    )
    actuator_score = min(1.0, state_delta / (0.05 + action_norm * 0.25))
    return {
        "image_mean": float(statistics.fmean(stats.get("mean", 0.0) for stats in image_stats))
        if image_stats
        else 0.0,
        "image_standard_deviation": std_mean,
        "image_zero_fraction": zero_fraction,
        "image_temporal_abs_difference": temporal_difference,
        "image_temporal_difference_range": temporal_range,
        "state_delta_norm": state_delta,
        "requested_action_norm": action_norm,
        "observation_evidence": observation_score,
        "actuator_evidence": actuator_score,
    }


class PassivePilotMonitor:
    """Fixed, non-trained evidence monitor used only for pilot diagnostics."""

    threshold = 0.75
    observation_diagnosis_threshold = 0.30

    def reset(self) -> None:
        return None

    def observe(self, context: MonitorContext) -> MonitorDecision:
        features = _monitor_features(context)
        observation_score = features["observation_evidence"]
        actuator_score = features["actuator_evidence"]
        failure_detected = max(observation_score, actuator_score) >= self.threshold
        failure_type = None
        if failure_detected:
            failure_type = (
                "observation_fault" if observation_score >= actuator_score else "actuator_fault"
            )
        return MonitorDecision(
            failure_detected=failure_detected,
            confidence=max(observation_score, actuator_score),
            failure_type=failure_type,
            evidence=features,
        )


class ContinueController:
    def reset(self) -> None:
        return None

    def decide(self, context: RecoveryContext) -> RecoveryDecision:
        del context
        return RecoveryDecision(RecoveryAction.CONTINUE, "pilot_fixed_continue")

    def execute(self, decision: RecoveryDecision) -> None:
        if decision.action is not RecoveryAction.CONTINUE:
            raise ValueError("pilot controller is fixed-continue")


def _jsonl_record(stream: Any, event_type: str, **payload: Any) -> None:
    from vla_recovery_bench.recording import to_jsonable

    stream.write(
        json.dumps(to_jsonable({"event_type": event_type, **payload}), sort_keys=True) + "\n"
    )
    stream.flush()


def _action_shapes(action: Any) -> Any:
    if isinstance(action, Mapping):
        return {str(key): _action_shapes(value) for key, value in action.items()}
    return list(getattr(action, "shape", ()))


def _run_episode(
    *,
    episode_index: int,
    item: Mapping[str, Any],
    environment: RoboCasaEnvironment,
    policy: GrootRoboCasaPolicy,
    client: TorchZmqPolicyClient,
    horizon: int,
    monitor_stream: Any,
    audit_stream: Any,
    episode_stream: Any,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(item["seed"])
    condition = str(item["condition"])
    episode_token = hashlib.sha256(
        f"v1.4-monitor-stream|{item['episode_id']}".encode()
    ).hexdigest()[:24]
    schedule = FaultSchedule(item["faults"])
    schedule.reset()
    seed_response = client.call("set_seed", {"seed": seed})
    if seed_response != {"seed": seed}:
        raise RuntimeError(f"policy server did not accept seed {seed}")
    policy.reset()
    observation = environment.reset(seed)
    monitor = PassivePilotMonitor()
    monitor.reset()
    recovery = ContinueController()
    recovery.reset()
    requested_history: list[Any] = []
    policy_latencies: list[float] = []
    monitor_latencies: list[float] = []
    fault_events: list[dict[str, Any]] = []
    alarms: list[dict[str, Any]] = []
    total_reward = 0.0
    success = False
    terminated = False
    truncated = False
    termination_reason = "horizon"
    started = time.perf_counter()
    chunk_length = int(getattr(policy, "action_chunk_length", 1))
    chunk_id = 0
    chunk_position = 0
    records_for_analysis: list[dict[str, Any]] = []
    observation_contract = {
        key: {
            "shape": list(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
        }
        for key, value in flatten_observation(observation).items()
    }
    action_contract: Any = None
    for step in range(horizon):
        for fault in schedule.due(step, FaultPhase.BEFORE_ACTION):
            application = environment.inject_fault(fault)
            event = {
                "episode_index": episode_index,
                "episode_id": item["episode_id"],
                "condition": condition,
                "seed": seed,
                "step": step,
                "phase": FaultPhase.BEFORE_ACTION,
                "fault": fault,
                "application": application,
                "injection_step": step,
                "first_affected_input_step": step,
            }
            fault_events.append(event)
            _jsonl_record(audit_stream, "fault_injection", **event)
        prompt = task_description(observation)
        action = policy.act(observation, prompt)
        action_contract = _action_shapes(action)
        if policy.last_inference_latency_ms is not None:
            policy_latencies.append(float(policy.last_inference_latency_ms))
            _jsonl_record(
                audit_stream,
                "policy_inference",
                episode_index=episode_index,
                episode_id=item["episode_id"],
                control_step=step,
                latency_ms=policy.last_inference_latency_ms,
                saturation=policy.last_chunk_saturation,
            )
        transition = environment.step(action)
        total_reward += transition.reward
        after_fault_applied = False
        for fault in schedule.due(step, FaultPhase.AFTER_STEP):
            application = environment.inject_fault(fault)
            after_fault_applied = after_fault_applied or application.applied
            first_affected = int(application.details.get("first_affected_input_step", step + 1))
            event = {
                "episode_index": episode_index,
                "episode_id": item["episode_id"],
                "condition": condition,
                "seed": seed,
                "step": step,
                "phase": FaultPhase.AFTER_STEP,
                "fault": fault,
                "application": application,
                "injection_step": step,
                "first_affected_input_step": first_affected,
            }
            fault_events.append(event)
            _jsonl_record(audit_stream, "fault_injection", **event)
        if after_fault_applied:
            apply_observation_fault = getattr(environment, "apply_pending_observation_fault", None)
            if not callable(apply_observation_fault):
                raise RuntimeError("RoboCasa environment lacks after-step observation fault hook")
            transition = replace(
                transition,
                observation=apply_observation_fault(transition.observation),
            )
        requested_history.append(action)
        requested_action_chunk = policy.requested_action_chunk
        if len(requested_action_chunk) != chunk_length:
            raise RuntimeError(
                "policy did not expose the complete requested action chunk: "
                f"expected={chunk_length}, got={len(requested_action_chunk)}"
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
            action_chunk=requested_action_chunk,
            chunk=chunk,
        )
        monitor_started = time.perf_counter()
        decision = monitor.observe(context)
        monitor_latencies.append((time.perf_counter() - monitor_started) * 1000.0)
        features = dict(decision.evidence)
        monitor_event = {
            "episode_token": episode_token,
            "control_step": step,
            "action_policy_input_step": step,
            "returned_observation_step": step + 1,
            "next_policy_input_step": step + 1,
            "requested_action": action,
            "requested_action_chunk": requested_action_chunk,
            "requested_action_history": tuple(requested_history[-chunk_length:]),
            "requested_action_shape": _action_shapes(action),
            "chunk": chunk,
            "decision": decision,
            "features": features,
        }
        _jsonl_record(monitor_stream, "monitor_step", **monitor_event)
        records_for_analysis.append(
            {
                "control_step": step,
                "observation_step": step + 1,
                **features,
                "alarm": decision.failure_detected,
            }
        )
        if decision.failure_detected:
            alarms.append(
                {
                    "control_step": step,
                    "observation_step": step + 1,
                    "failure_type": decision.failure_type,
                }
            )
        recovery_decision = recovery.decide(
            RecoveryContext(
                episode_id=0,
                step=step + 1,
                instruction=prompt,
                observation=transition.observation,
                monitor=decision,
                retry_count=0,
                chunk=chunk,
            )
        )
        recovery.execute(recovery_decision)
        _jsonl_record(
            audit_stream,
            "audit_transition",
            episode_index=episode_index,
            episode_id=item["episode_id"],
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
    steps = step + 1
    for event in fault_events:
        event["exposed"] = bool(steps > int(event["first_affected_input_step"]))
    exposed = [event for event in fault_events if event["exposed"]]
    first_affected = min(
        (int(event["first_affected_input_step"]) for event in exposed),
        default=None,
    )
    exposure_duration = (
        int(exposed[0]["application"].details.get("duration", 0)) if exposed else None
    )
    exposure_end = (
        first_affected + exposure_duration
        if first_affected is not None and exposure_duration is not None
        else None
    )
    detection_step_key = "observation_step" if condition == "observation_fault" else "control_step"
    matched_alarm = next(
        (
            alarm
            for alarm in alarms
            if (
                first_affected is not None
                and exposure_end is not None
                and first_affected <= alarm[detection_step_key] < exposure_end
            )
        ),
        None,
    )
    detection_delay = (
        int(matched_alarm[detection_step_key] - first_affected)
        if matched_alarm is not None
        else None
    )
    summary = {
        "episode_index": episode_index,
        "episode_id": item["episode_id"],
        "pair_id": item["pair_id"],
        "seed": seed,
        "condition": condition,
        "mechanism": "none" if condition == "clean" else condition,
        "monitor_episode_token": episode_token,
        "configured_faults": item["faults"],
        "configured_fault_count": len(item["faults"]),
        "applied_fault_count": len(fault_events),
        "exposed_fault_count": len(exposed),
        "not_exposed": bool(condition != "clean" and not exposed),
        "first_affected_input_step": first_affected,
        "exposure_end_step_exclusive": exposure_end,
        "alarm_count": len(alarms),
        "first_alarm_control_step": alarms[0]["control_step"] if alarms else None,
        "first_alarm_observation_step": (alarms[0]["observation_step"] if alarms else None),
        "first_exposure_alarm_control_step": (
            matched_alarm["control_step"] if matched_alarm else None
        ),
        "first_exposure_alarm_observation_step": (
            matched_alarm["observation_step"] if matched_alarm else None
        ),
        "detection_delay_steps": detection_delay,
        "detection_reference_step": detection_step_key,
        "success": success,
        "steps": steps,
        "reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": termination_reason,
        "wall_time_seconds": time.perf_counter() - started,
        "task_id": provenance["task_id"],
        "split": provenance["split"],
        "policy_provenance": provenance["policy"],
        "environment_provenance": provenance["environment"],
        "observation_contract": observation_contract,
        "action_contract": {
            "shape": action_contract,
            "all_steps_finite_and_in_range": True,
        },
        "policy_inference_count": len(policy_latencies),
        "policy_inference_latency_ms": {
            "mean": statistics.fmean(policy_latencies) if policy_latencies else None,
            "p95": float(np.percentile(policy_latencies, 95)) if policy_latencies else None,
            "maximum": max(policy_latencies) if policy_latencies else None,
        },
        "monitor_latency_ms": {
            "mean": statistics.fmean(monitor_latencies) if monitor_latencies else None,
            "p95": (float(np.percentile(monitor_latencies, 95)) if monitor_latencies else None),
            "maximum": max(monitor_latencies) if monitor_latencies else None,
        },
        "monitor_output": {
            "calibrated": False,
            "first_diagnosed_type": alarms[0]["failure_type"] if alarms else None,
            "mean_observation_evidence": statistics.fmean(
                item["observation_evidence"] for item in records_for_analysis
            ),
            "maximum_observation_evidence": max(
                item["observation_evidence"] for item in records_for_analysis
            ),
            "mean_actuator_evidence": statistics.fmean(
                item["actuator_evidence"] for item in records_for_analysis
            ),
            "maximum_actuator_evidence": max(
                item["actuator_evidence"] for item in records_for_analysis
            ),
        },
        "selected_intervention": {
            "action": RecoveryAction.CONTINUE,
            "reason": "pilot_fixed_continue",
        },
        "action_saturated_values": int(policy.episode_saturated_values),
        "analysis_steps": records_for_analysis,
    }
    persisted_summary = {key: value for key, value in summary.items() if key != "analysis_steps"}
    _jsonl_record(episode_stream, "episode", **persisted_summary)
    return summary


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_condition[str(result["condition"])].append(result)
    condition_metrics: dict[str, Any] = {}
    for condition, rows in by_condition.items():
        delays = [
            row["detection_delay_steps"] for row in rows if row["detection_delay_steps"] is not None
        ]
        condition_metrics[condition] = {
            "episodes": len(rows),
            "successes": sum(bool(row["success"]) for row in rows),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "mean_steps": statistics.fmean(row["steps"] for row in rows),
            "mean_reward": statistics.fmean(row["reward"] for row in rows),
            "faults_configured": sum(row["configured_fault_count"] for row in rows),
            "faults_applied": sum(row["applied_fault_count"] for row in rows),
            "faults_exposed": sum(row["exposed_fault_count"] for row in rows),
            "not_exposed_episodes": sum(bool(row["not_exposed"]) for row in rows),
            "false_alarm_episodes": sum(
                bool(row["condition"] == "clean" and row["alarm_count"] > 0) for row in rows
            ),
            "exposure_window_detections": sum(
                bool(row["exposed_fault_count"] and row["detection_delay_steps"] is not None)
                for row in rows
            ),
            "mean_detection_delay_steps": statistics.fmean(delays) if delays else None,
        }
    fault_rows = [
        row for row in results if row["condition"] in {"actuator_fault", "observation_fault"}
    ]
    exposed_rows = [row for row in fault_rows if row["exposed_fault_count"] > 0]
    predictions: list[tuple[str, str]] = []
    for row in exposed_rows:
        steps = row["analysis_steps"]
        first = row["first_affected_input_step"]
        end = row["exposure_end_step_exclusive"]
        step_key = "observation_step" if row["condition"] == "observation_fault" else "control_step"
        window = [
            item
            for item in steps
            if first is not None and end is not None and first <= item[step_key] < end
        ]
        if not window:
            continue
        obs = max(item["observation_evidence"] for item in window)
        predicted = (
            "observation_fault"
            if obs >= PassivePilotMonitor.observation_diagnosis_threshold
            else "actuator_fault"
        )
        predictions.append((str(row["condition"]), predicted))
    confusion = Counter((actual, predicted) for actual, predicted in predictions)
    balanced_accuracy = None
    if predictions:
        recalls = []
        for label in ("actuator_fault", "observation_fault"):
            positives = [pair for pair in predictions if pair[0] == label]
            recalls.append(
                sum(actual == predicted for actual, predicted in positives) / len(positives)
                if positives
                else 0.0
            )
        balanced_accuracy = statistics.fmean(recalls)
    return {
        "episodes": len(results),
        "conditions": condition_metrics,
        "exploratory_passive_rule": {
            "name": "episode_max_observation_evidence_threshold",
            "exposed_fault_episodes": len(predictions),
            "confusion": {
                f"{actual}->{predicted}": count
                for (actual, predicted), count in sorted(confusion.items())
            },
            "balanced_accuracy": balanced_accuracy,
            "reference": 0.5,
            "minimum_useful_gain": 0.15,
            "scientific_result": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scene-seeds", type=int, default=None)
    parser.add_argument("--environment-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="target", choices=("target",))
    parser.add_argument("--horizon", type=int, default=750)
    parser.add_argument("--policy-python", default="/home/pc/VLA/envs/groot/bin/python")
    parser.add_argument("--server-script", default=str(ROOT / "scripts/serve_groot_policy.py"))
    parser.add_argument("--port", type=int, default=5566)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--server-start-timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("source /home/pc/VLA/env.sh before running the pilot")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if args.max_scene_seeds is not None and args.max_scene_seeds <= 0:
        raise ValueError("--max-scene-seeds must be positive")
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    plan = pilot_episode_plan(config)
    all_seeds = list(config["seed_block"]["seeds"])
    if args.max_scene_seeds is not None:
        selected_seeds = all_seeds[: args.max_scene_seeds]
        plan = [item for item in plan if item["seed"] in selected_seeds]
    else:
        selected_seeds = all_seeds
    expected_count = len(selected_seeds) * 3
    if len(plan) != expected_count:
        raise RuntimeError(f"pilot plan has {len(plan)} episodes; expected {expected_count}")
    plan = sorted(
        plan,
        key=lambda item: hashlib.sha256(
            f"{config['protocol_version']}|{item['episode_id']}".encode()
        ).hexdigest(),
    )
    output_dir = ensure_empty_output_dir(args.output)
    checkpoint_hashes = _verify_checkpoint(manifest, args.checkpoint)
    debug = args.max_scene_seeds is not None
    artifact_paths = {
        "run_manifest": output_dir / "run_manifest.json",
        "episodes": output_dir / "episodes.jsonl",
        "monitor_stream": output_dir / "monitor_stream.jsonl",
        "audit_stream": output_dir / "audit_stream.jsonl",
        "metrics": output_dir / "metrics.json",
        "monitor_config": output_dir / "monitor_config.json",
        "calibration": output_dir / "calibration.json",
        "software_versions": output_dir / "software_versions.json",
        "policy_state_before": output_dir / "policy_state_before.json",
        "policy_state_after": output_dir / "policy_state_after.json",
        "artifact_validation": output_dir / "artifact_validation.json",
    }
    write_json_once(
        artifact_paths["monitor_config"],
        {
            "name": "passive_pilot_evidence_monitor",
            "version": "1.0",
            "trained": False,
            "inputs": [
                "previous_observation",
                "current_observation",
                "requested_action_history",
                "task_instruction",
            ],
            "forbidden_inputs": [
                "reward",
                "info",
                "success",
                "fault_schedule",
                "executed_action",
                "mujoco_state",
            ],
            "alarm_threshold": PassivePilotMonitor.threshold,
            "observation_diagnosis_threshold": (
                PassivePilotMonitor.observation_diagnosis_threshold
            ),
            "execution_order": "sha256(protocol_version|episode_id)",
            "online_episode_id": "reset-local zero; no condition-coded global index",
            "scientific_role": "evaluation_only_identifiability_diagnostic",
        },
    )
    write_json_once(
        artifact_paths["calibration"],
        {"status": "not_fitted", "reason": "pilot is not a calibration result"},
    )
    write_json_once(
        artifact_paths["software_versions"],
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
        write_json_once(artifact_paths["policy_state_before"], policy_state_before)
        environment = RoboCasaEnvironment(args.environment_id, split=args.split)
        policy = GrootRoboCasaPolicy(environment.action_space, client)
        provenance = {
            "task_id": args.environment_id,
            "split": args.split,
            "policy": {
                "name": manifest["policy_name"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "manifest": str(args.manifest.resolve()),
            },
            "environment": {
                "repository_commit": _git_commit(ROOT),
                "robocasa_commit": _git_commit("/home/pc/VLA/src/robocasa"),
                "robosuite_commit": _git_commit("/home/pc/VLA/src/robosuite"),
                "groot_commit": _git_commit("/home/pc/VLA/src/Isaac-GR00T"),
            },
        }
        with (
            artifact_paths["episodes"].open("x", encoding="utf-8") as episode_stream,
            artifact_paths["monitor_stream"].open("x", encoding="utf-8") as monitor_stream,
            artifact_paths["audit_stream"].open("x", encoding="utf-8") as audit_stream,
        ):
            for index, item in enumerate(plan):
                result = _run_episode(
                    episode_index=index,
                    item=item,
                    environment=environment,
                    policy=policy,
                    client=client,
                    horizon=args.horizon,
                    monitor_stream=monitor_stream,
                    audit_stream=audit_stream,
                    episode_stream=episode_stream,
                    provenance=provenance,
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "episode_index",
                                "episode_id",
                                "condition",
                                "seed",
                                "success",
                                "steps",
                                "configured_fault_count",
                                "applied_fault_count",
                                "exposed_fault_count",
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
            raise RuntimeError("frozen GR00T parameter hash changed during pilot")
        write_json_once(artifact_paths["policy_state_after"], policy_state_after)
        metrics = {
            "status": "completed",
            "scientific_result": False,
            "protocol_version": config["protocol_version"],
            "debug": debug,
            "environment": {
                "id": args.environment_id,
                "split": args.split,
                "horizon": args.horizon,
                "control_frequency_hz": config["environment"]["control_frequency_hz"],
                "action_chunk_length": config["environment"]["action_chunk_length"],
                "action_space": describe_action_space(environment.action_space),
            },
            "policy": {
                "name": manifest["policy_name"],
                "manifest": str(args.manifest.resolve()),
                "checkpoint": str(args.checkpoint),
                "checkpoint_files_sha256": checkpoint_hashes,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "parameter_sha256_before": policy_state_before["current_parameter_sha256"],
                "parameter_sha256_after": policy_state_after["current_parameter_sha256"],
                "frozen": True,
            },
            "seeds": selected_seeds,
            "episode_count": len(results),
            "summary": _summarize(results),
            "outputs": {key: str(path) for key, path in artifact_paths.items()},
        }
        write_json_once(artifact_paths["metrics"], metrics)
        write_json_once(
            artifact_paths["run_manifest"],
            {
                "protocol_version": config["protocol_version"],
                "status": "completed",
                "scientific_result": False,
                "debug": debug,
                "environment": metrics["environment"],
                "policy": metrics["policy"],
                "monitor": _load_json(artifact_paths["monitor_config"]),
                "seeds": selected_seeds,
                "episode_plan": plan,
                "timing_contract": config["timing_contract"],
                "config": {
                    "path": str(args.config.resolve()),
                    "sha256": _sha256(args.config),
                },
                "policy_manifest": {
                    "path": str(args.manifest.resolve()),
                    "sha256": _sha256(args.manifest),
                },
                "command": [sys.executable, *sys.argv],
                "artifacts": metrics["outputs"],
            },
        )
        artifact_errors = validate_pilot_artifacts(
            output_dir,
            expected_episode_count=expected_count,
        )
        write_json_once(
            artifact_paths["artifact_validation"],
            {
                "status": "passed" if not artifact_errors else "failed",
                "expected_episode_count": expected_count,
                "errors": artifact_errors,
            },
        )
        if artifact_errors:
            raise RuntimeError(f"pilot artifact validation failed: {artifact_errors}")
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
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
