#!/usr/bin/env python3
"""Run the frozen v1.1 RoboCasa diagnostic-probe evaluation with v1.2 relock.

The runner keeps the online monitor channel separate from the privileged audit
channel.  It supports a three-seed debug run and the pre-registered 12-seed
pilot; final-test seeds require a separately issued lock artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vla_recovery_bench.artifacts import ensure_empty_output_dir, write_json_once
from vla_recovery_bench.diagnostic_probe import (
    FORBIDDEN_ONLINE_FIELDS,
    MAX_PROBE_STEPS,
    assert_online_event_safe,
    monitor_parameter_sha256,
    trigger_from_prediction,
)
from vla_recovery_bench.faults import FaultSchedule
from vla_recovery_bench.groot_adapter import (
    CAMERA_SHAPES,
    GrootRoboCasaPolicy,
    TorchZmqPolicyClient,
    flatten_observation,
    task_description,
)
from vla_recovery_bench.monitor import MECHANISMS, FaultConditionedTemporalMonitor
from vla_recovery_bench.monitor_protocol import (
    validate_monitor_relock_protocol,
    validate_probe_protocol,
)
from vla_recovery_bench.pilot import pilot_episode_plan
from vla_recovery_bench.recording import to_jsonable
from vla_recovery_bench.robocasa_adapter import RoboCasaEnvironment, action_shape
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
except ModuleNotFoundError:  # pragma: no cover - direct script execution
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

DEFAULT_PILOT_CONFIG = ROOT / "configs/identifiability_pilot_v1_4.json"
DEFAULT_PROBE_CONFIG = ROOT / "configs/diagnostic_probe_v1_1.json"
DEFAULT_MONITOR = Path("/home/pc/VLA/outputs/monitor_v1_0_formal_model/monitor.npz")
DEFAULT_MONITOR_METRICS = Path("/home/pc/VLA/outputs/monitor_v1_0_formal_model/metrics.json")
DEFAULT_MONITOR_PROTOCOL = ROOT / "configs/monitor_relock_v1_2.json"
DEFAULT_LOCK = Path("/home/pc/VLA/outputs/diagnostic_probe_v1_2_lock/probe_lock.json")


def _stable_hash(value: Any) -> str:
    """Hash nested observations/actions without lossy numeric conversion."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
            digest.update(b"}")
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"array")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
            return
        if isinstance(item, np.generic):
            visit(item.item())
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence[")
            for child in item:
                visit(child)
            digest.update(b"]")
            return
        digest.update(type(item).__name__.encode("ascii", errors="replace"))
        digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _jsonl(stream: Any, event_type: str, **payload: Any) -> None:
    event = {"event_type": event_type, **payload}
    assert_online_event_safe(event) if event_type in {"monitor_step", "probe_step"} else None
    stream.write(json.dumps(to_jsonable(event), sort_keys=True) + "\n")
    stream.flush()


def _observation_contract(observation: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in flatten_observation(observation).items():
        array = np.asarray(value)
        entry: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if array.dtype.kind in "biufc" and array.size:
            entry["minimum"] = float(np.min(array))
            entry["maximum"] = float(np.max(array))
        result[key] = entry
    return result


def _delta_stats(previous: Any, current: Any) -> dict[str, float] | None:
    """Return explicit numeric delta statistics without changing either value."""
    previous_array = np.asarray(previous)
    current_array = np.asarray(current)
    if previous_array.shape != current_array.shape:
        return None
    if previous_array.dtype.kind not in "biufc" or current_array.dtype.kind not in "biufc":
        return None
    old = previous_array.astype(np.float32, copy=False)
    new = current_array.astype(np.float32, copy=False)
    if np.issubdtype(previous_array.dtype, np.integer):
        old = old / 255.0
    if np.issubdtype(current_array.dtype, np.integer):
        new = new / 255.0
    difference = np.abs(new - old)
    return {
        "mean_absolute": float(difference.mean()) if difference.size else 0.0,
        "maximum_absolute": float(difference.max()) if difference.size else 0.0,
        "l2": float(np.linalg.norm(new - old)),
    }


def _observation_delta(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Summarize shape-preserving numeric deltas, separating camera leaves."""
    old = flatten_observation(previous)
    new = flatten_observation(current)
    deltas: dict[str, dict[str, float]] = {}
    camera_values: list[float] = []
    for key in sorted(set(old) & set(new)):
        stats = _delta_stats(old[key], new[key])
        if stats is None:
            continue
        deltas[key] = stats
        value = np.asarray(new[key])
        if value.ndim == 3 and value.shape[-1] in (3, 4):
            camera_values.append(stats["mean_absolute"])
    mean_camera_delta = (
        float(statistics.fmean(camera_values)) if camera_values else 0.0
    )
    camera_summary = {
        "mean_absolute_difference": mean_camera_delta,
        "temporal_consistency": float(np.clip(1.0 - mean_camera_delta, 0.0, 1.0)),
        "camera_count": len(camera_values),
    }
    return deltas, camera_summary


def _action_delta(previous: Any | None, current: Any) -> dict[str, dict[str, float]]:
    """Summarize requested-action changes for the intervention audit."""
    if previous is None:
        return {}
    old = flatten_observation(previous) if isinstance(previous, Mapping) else {"action": previous}
    new = flatten_observation(current) if isinstance(current, Mapping) else {"action": current}
    result: dict[str, dict[str, float]] = {}
    for key in sorted(set(old) & set(new)):
        stats = _delta_stats(old[key], new[key])
        if stats is not None:
            result[key] = stats
    return result


def _posterior_delta(
    baseline: Mapping[str, Any] | None, posterior: Mapping[str, Any]
) -> dict[str, float]:
    """Compare a probe posterior with the first trigger posterior."""
    if baseline is None or not isinstance(baseline.get("posterior"), Mapping):
        return {}
    return {
        name: float(posterior[name]) - float(baseline["posterior"][name])
        for name in MECHANISMS
    }


def _chunk_metadata(
    policy: GrootRoboCasaPolicy,
    *,
    horizon: int,
    step: int,
    probe_active: bool,
) -> ActionChunkMetadata:
    state = policy.chunk_state()
    return ActionChunkMetadata(
        chunk_id=int(state["chunk_id"]),
        position_in_chunk=int(state["position_in_chunk"]),
        chunk_length=int(state["chunk_length"]),
        remaining_horizon=max(horizon - step - 1, 0),
        policy_inference_latency_ms=(
            None
            if state.get("policy_inference_latency_ms") is None
            else float(state["policy_inference_latency_ms"])
        ),
        camera_keys=tuple(CAMERA_SHAPES),
        probe_active=probe_active,
    )


def _fault_event(
    *,
    episode_index: int,
    item: Mapping[str, Any],
    arm: str,
    seed: int,
    step: int,
    phase: FaultPhase,
    fault: Any,
    application: Any,
) -> dict[str, Any]:
    return {
        "episode_index": episode_index,
        "episode_id": item["episode_id"],
        "pair_id": item["pair_id"],
        "condition": item["condition"],
        "arm": arm,
        "seed": seed,
        "control_step": step,
        "phase": phase,
        "fault": fault,
        "application": application,
    }


def run_episode(
    *,
    episode_index: int,
    item: Mapping[str, Any],
    arm: str,
    environment: RoboCasaEnvironment,
    policy: GrootRoboCasaPolicy,
    client: Any,
    monitor: FaultConditionedTemporalMonitor,
    risk_threshold: float,
    entropy_threshold: float,
    base_horizon: int,
    monitor_stream: Any,
    probe_stream: Any,
    audit_stream: Any,
    episode_stream: Any,
) -> dict[str, Any]:
    """Run one episode; labels remain confined to the audit-side arguments."""
    seed = int(item["seed"])
    token = hashlib.sha256(
        f"diagnostic-probe-v1.1|{item['episode_id']}|{arm}".encode()
    ).hexdigest()[:32]
    schedule = FaultSchedule(item.get("faults", ()))
    schedule.reset()
    response = client.call("set_seed", {"seed": seed})
    if response != {"seed": seed}:
        raise RuntimeError(f"policy server did not accept seed {seed}")
    policy.reset()
    monitor.reset()
    observation = environment.reset(seed)
    prompt = task_description(observation)
    first_contract = _observation_contract(observation)
    prefix_parts: list[str] = []
    monitor_records: list[dict[str, Any]] = []
    alarms: list[dict[str, Any]] = []
    fault_events: list[dict[str, Any]] = []
    policy_latencies: list[float] = []
    probe_latencies: list[float] = []
    probe_actions: list[dict[str, Any]] = []
    previous_requested_action: Any | None = None
    trigger_observation_step: int | None = None
    post_window_observation_step: int | None = None
    post_window_posterior: dict[str, float] | None = None
    trigger_prediction: dict[str, Any] | None = None
    probe_offset: int | None = None
    probe_started_step: int | None = None
    probe_finished_step: int | None = None
    total_reward = 0.0
    success = False
    terminated = False
    truncated = False
    termination_reason = "horizon"
    step = 0
    action_contract: Any = None

    while True:
        if probe_offset is None and step >= base_horizon:
            break
        if probe_offset is not None and step >= base_horizon + MAX_PROBE_STEPS:
            break

        for fault in schedule.due(step, FaultPhase.BEFORE_ACTION):
            application = environment.inject_fault(fault)
            if not application.applied:
                raise RuntimeError(f"declared fault was not applied: {application}")
            event = _fault_event(
                episode_index=episode_index,
                item=item,
                arm=arm,
                seed=seed,
                step=step,
                phase=FaultPhase.BEFORE_ACTION,
                fault=fault,
                application=application,
            )
            fault_events.append(event)
            _jsonl(audit_stream, "fault_injection", **event)

        active_probe_offset = probe_offset
        if arm == "passive_plus_probe" and active_probe_offset is not None:
            if active_probe_offset == 0:
                action = policy.repeat_last_action()
                operation = "repeat_previous_requested_action"
                if probe_started_step is None:
                    probe_started_step = step
            elif active_probe_offset == 1:
                policy.force_requery()
                action = policy.act(observation, prompt)
                operation = "force_requery_and_execute_first_action"
            else:
                action = policy.act(observation, prompt)
                operation = "continue_requeried_chunk"
            probe_actions.append(
                {
                    "probe_step": active_probe_offset,
                    "control_step": step,
                    "operation": operation,
                }
            )
        else:
            action = policy.act(observation, prompt)

        action_contract = action_shape(action)
        if policy.last_inference_latency_ms is not None:
            latency = float(policy.last_inference_latency_ms)
            policy_latencies.append(latency)
            if active_probe_offset is not None:
                probe_latencies.append(latency)

        transition = environment.step(action)
        total_reward += float(transition.reward)
        after_fault_applied = False
        for fault in schedule.due(step, FaultPhase.AFTER_STEP):
            application = environment.inject_fault(fault)
            if not application.applied:
                raise RuntimeError(f"declared fault was not applied: {application}")
            after_fault_applied = True
            event = _fault_event(
                episode_index=episode_index,
                item=item,
                arm=arm,
                seed=seed,
                step=step,
                phase=FaultPhase.AFTER_STEP,
                fault=fault,
                application=application,
            )
            fault_events.append(event)
            _jsonl(audit_stream, "fault_injection", **event)
        if after_fault_applied:
            transition = replace(
                transition,
                observation=environment.apply_pending_observation_fault(transition.observation),
            )

        requested_chunk = policy.requested_action_chunk
        if len(requested_chunk) != policy.action_chunk_length:
            raise RuntimeError("policy did not expose a complete requested action chunk")
        chunk = _chunk_metadata(
            policy,
            horizon=base_horizon,
            step=step,
            probe_active=active_probe_offset is not None,
        )
        context = MonitorContext(
            episode_id=episode_index,
            step=step + 1,
            instruction=prompt,
            previous_observation=observation,
            observation=transition.observation,
            action=action,
            action_chunk=requested_chunk,
            chunk=chunk,
        )
        decision = monitor.observe(context)
        prediction = {
            "risk": float(decision.evidence["risk"]),
            "posterior": dict(decision.evidence["posterior"]),
            "normalized_entropy": float(decision.evidence["normalized_entropy"]),
            "predicted_mechanism": str(decision.evidence.get("predicted_mechanism", "none")),
        }
        trigger = trigger_from_prediction(
            prediction,
            risk_threshold=risk_threshold,
            entropy_threshold=entropy_threshold,
        )
        trigger_was_unset = trigger_observation_step is None
        if trigger_was_unset and trigger["joint_trigger"]:
            trigger_observation_step = step + 1
            post_window_observation_step = trigger_observation_step + MAX_PROBE_STEPS
            trigger_prediction = {
                **trigger,
                "posterior": {
                    name: float(prediction["posterior"][name]) for name in MECHANISMS
                },
                "predicted_mechanism": prediction["predicted_mechanism"],
            }
            if arm == "passive_plus_probe":
                probe_offset = 0
                probe_started_step = step + 1
        if trigger["joint_trigger"]:
            alarms.append(
                {
                    "observation_step": step + 1,
                    "control_step": step,
                    "risk_alarm": trigger["risk_alarm"],
                    "entropy_alarm": trigger["entropy_alarm"],
                    "joint_trigger": trigger["joint_trigger"],
                }
            )

        current_hash = _stable_hash(transition.observation)
        action_hash = _stable_hash(action)
        if trigger_was_unset and not trigger["joint_trigger"]:
            prefix_parts.extend((current_hash, action_hash))

        monitor_event = {
            "episode_token": token,
            "control_step": step,
            "observation_step": step + 1,
            "requested_action": action,
            "requested_action_chunk": requested_chunk,
            "chunk": chunk,
            "posterior": prediction["posterior"],
            "risk": prediction["risk"],
            "normalized_entropy": prediction["normalized_entropy"],
            "risk_alarm": trigger["risk_alarm"],
            "entropy_alarm": trigger["entropy_alarm"],
            "joint_trigger": trigger["joint_trigger"],
            "probe_active": active_probe_offset is not None,
            "probe_step": active_probe_offset,
        }
        _jsonl(monitor_stream, "monitor_step", **monitor_event)
        monitor_records.append(monitor_event)
        if (
            post_window_observation_step is not None
            and step + 1 == post_window_observation_step
        ):
            post_window_posterior = {
                name: float(prediction["posterior"][name]) for name in MECHANISMS
            }

        _jsonl(
            audit_stream,
            "audit_transition",
            episode_index=episode_index,
            episode_id=item["episode_id"],
            episode_token=token,
            pair_id=item["pair_id"],
            condition=item["condition"],
            arm=arm,
            seed=seed,
            control_step=step,
            probe_step=active_probe_offset,
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

        if active_probe_offset is not None:
            observation_delta, camera_consistency = _observation_delta(
                observation, transition.observation
            )
            _jsonl(
                probe_stream,
                "probe_step",
                episode_token=token,
                control_step=step,
                observation_step=step + 1,
                probe_step=active_probe_offset,
                operation=probe_actions[-1]["operation"],
                requested_action=action,
                requested_action_hash=action_hash,
                posterior=prediction["posterior"],
                risk=prediction["risk"],
                normalized_entropy=prediction["normalized_entropy"],
                policy_inference_latency_ms=policy.last_inference_latency_ms,
                observation_delta=observation_delta,
                requested_action_delta=_action_delta(previous_requested_action, action),
                camera_temporal_consistency=camera_consistency,
                mechanism_posterior_delta=_posterior_delta(
                    trigger_prediction, prediction["posterior"]
                ),
            )

        observation = transition.observation
        previous_requested_action = action
        prompt = task_description(observation)
        success = bool(transition.info.get("success", False))
        terminated = bool(transition.terminated)
        truncated = bool(transition.truncated)
        step += 1

        if active_probe_offset is not None:
            if active_probe_offset == MAX_PROBE_STEPS - 1:
                probe_finished_step = step
                policy.force_requery()
                probe_offset = None
            else:
                probe_offset = active_probe_offset + 1

        if success:
            termination_reason = "success"
            break
        if terminated or truncated:
            termination_reason = "environment_done"
            break

    if trigger_observation_step is not None and probe_started_step is None:
        # The alarm occurred at the final available transition, or the episode
        # terminated before the first probe action. Retain explicit attrition.
        probe_finished_step = None
    steps = step
    summary = {
        "episode_index": episode_index,
        "episode_id": item["episode_id"],
        "pair_id": item["pair_id"],
        "episode_token": token,
        "arm": arm,
        "condition": item["condition"],
        "mechanism": (
            "none" if item["condition"] == "clean" else item["condition"]
        ),
        "seed": seed,
        "steps": steps,
        "base_horizon": base_horizon,
        "success": success,
        "reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": termination_reason,
        "configured_faults": item.get("faults", ()),
        "configured_fault_count": len(item.get("faults", ())),
        "applied_fault_count": len(fault_events),
        "alarms": alarms,
        "triggered": trigger_observation_step is not None,
        "trigger_observation_step": trigger_observation_step,
        "post_window_observation_step": post_window_observation_step,
        "post_window_posterior": post_window_posterior,
        "trigger_prediction": trigger_prediction,
        "probe_executed": bool(probe_actions),
        "probe_steps": len(probe_actions),
        "probe_started_step": probe_started_step,
        "probe_finished_step": probe_finished_step,
        "probe_complete": len(probe_actions) == MAX_PROBE_STEPS,
        "probe_compute_ms": float(sum(probe_latencies)),
        "probe_requery_count": sum(
            1
            for row in probe_actions
            if row["operation"] == "force_requery_and_execute_first_action"
        ),
        "max_probe_steps": MAX_PROBE_STEPS,
        "prefix_hash_to_trigger": _stable_hash(prefix_parts),
        "observation_contract": first_contract,
        "action_contract": {"shape": action_contract, "finite_and_in_range": True},
        "policy_inference_count": len(policy_latencies),
        "policy_inference_latency_ms": {
            "mean": statistics.fmean(policy_latencies) if policy_latencies else None,
            "p95": float(np.percentile(policy_latencies, 95)) if policy_latencies else None,
            "maximum": max(policy_latencies) if policy_latencies else None,
        },
        "fault_events": fault_events,
        "monitor_record_count": len(monitor_records),
        "monitor_parameter_sha256": monitor_parameter_sha256(monitor),
    }
    _jsonl(episode_stream, "episode", **summary)
    return summary


def _plan_for_stage(
    config: Mapping[str, Any], stage: str, *, final_seeds: Sequence[int] = ()
) -> tuple[list[int], list[dict[str, Any]]]:
    all_seeds = [int(seed) for seed in config["seed_block"]["seeds"]]
    if stage == "debug":
        selected = all_seeds[:3]
    elif stage == "pilot":
        selected = all_seeds
    else:
        selected = [int(seed) for seed in final_seeds]
    base = [row for row in pilot_episode_plan(config) if int(row["seed"]) in set(selected)]
    rows: list[dict[str, Any]] = []
    for item in base:
        for arm in ("passive_only", "passive_plus_probe"):
            rows.append({**item, "arm": arm})
    rows.sort(
        key=lambda item: hashlib.sha256(
            f"diagnostic-probe-v1.1|{stage}|{item['episode_id']}|{item['arm']}".encode()
        ).hexdigest()
    )
    return selected, rows


def _validate_lock(
    lock_path: Path,
    *,
    monitor_path: Path,
    probe_path: Path,
    risk_threshold: float,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    lock = _load_json(lock_path)
    if lock.get("status") != "locked":
        raise ValueError("probe lock is not in locked state")
    if lock.get("probe_protocol_version") != "1.1":
        raise ValueError("probe lock protocol version mismatch")
    if protocol_path is not None:
        if str(lock.get("protocol", {}).get("sha256")) != _sha256(protocol_path):
            raise ValueError("probe lock monitor protocol SHA256 does not match protocol")
        protocol = _load_json(protocol_path)
        relock_version = str(protocol.get("relock_version", ""))
        if relock_version not in {"1.2", "1.3"}:
            raise ValueError(
                "diagnostic probe requires a supported v1.2 or v1.3 monitor relock protocol"
            )
        reference = Path(str(protocol.get("parent_monitor_protocol", "")))
        candidates = [
            protocol_path.parent / reference,
            protocol_path.parents[1] / reference,
            reference,
        ]
        parent_path = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()), None
        )
        if parent_path is None:
            raise ValueError("monitor relock parent protocol is missing")
        parent = _load_json(parent_path)
        relock_errors = validate_monitor_relock_protocol(
            protocol,
            parent_config=parent,
            parent_sha256=_sha256(parent_path),
        )
        if relock_errors:
            raise ValueError(f"invalid monitor relock protocol: {relock_errors}")
        if lock.get("relock", {}).get("version") != relock_version:
            raise ValueError(
                "probe lock relock version does not match the monitor protocol"
            )
    if str(lock.get("monitor", {}).get("sha256")) != _sha256(monitor_path):
        raise ValueError("probe lock monitor SHA256 does not match checkpoint")
    if str(lock.get("probe_protocol", {}).get("sha256")) != _sha256(probe_path):
        raise ValueError("probe lock config SHA256 does not match probe config")
    locked_risk = float(lock.get("monitor", {}).get("risk_threshold"))
    if not np.isclose(locked_risk, risk_threshold, rtol=0.0, atol=1e-12):
        raise ValueError("probe lock risk threshold does not match monitor checkpoint")
    entropy = float(lock.get("calibration", {}).get("entropy_threshold"))
    if not np.isfinite(entropy):
        raise ValueError("probe lock entropy threshold is not finite")
    calibration = lock.get("calibration", {})
    validation = lock.get("validation", {})
    if int(calibration.get("episode_count", -1)) != 50:
        raise ValueError("probe lock must be based on 50 clean calibration episodes")
    if int(validation.get("clean_episode_count", -1)) != 50:
        raise ValueError("probe lock must validate 50 clean holdout episodes")
    try:
        validation_rate = float(validation["joint_trigger_rate"])
        max_rate = float(validation["max_union_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("probe lock is missing validation clean-budget fields") from error
    if not np.isfinite(validation_rate) or not np.isfinite(max_rate):
        raise ValueError("probe lock validation rates must be finite")
    if validation_rate > max_rate:
        raise ValueError("probe lock validation clean budget is exceeded")
    if protocol_path is not None:
        interval = lock.get("rate_reports", {}).get("validation_joint_trigger", {}).get(
            "clopper_pearson_95_percent"
        )
        if not isinstance(interval, Mapping):
            raise ValueError("probe lock is missing its confidence interval report")
        for bound in ("lower", "upper"):
            if not np.isfinite(float(interval.get(bound, np.nan))):
                raise ValueError("probe lock confidence interval is not finite")
    if lock.get("formula", {}).get("threshold_locked_before_pilot") is not True:
        raise ValueError("probe lock is missing the before-pilot threshold lock assertion")
    artifact_path = lock_path.parent / "artifact_validation.json"
    if not artifact_path.is_file():
        raise ValueError(f"missing lock artifact validation: {artifact_path}")
    artifact = _load_json(artifact_path)
    if artifact.get("status") != "passed":
        raise ValueError("probe lock artifact validation is not passed")
    declared_lock_hash = artifact.get("lock_sha256")
    if declared_lock_hash and str(declared_lock_hash) != _sha256(lock_path):
        raise ValueError("probe lock artifact validation hash does not match lock")
    return lock


def _validate_run_artifacts(
    output: Path,
    expected_count: int,
    *,
    stage: str,
    expected_plan: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    errors: list[str] = []
    required = (
        "run_manifest.json",
        "episodes.jsonl",
        "monitor_stream.jsonl",
        "probe_stream.jsonl",
        "privileged_audit.jsonl",
        "metrics.json",
        "monitor_config.json",
        "calibration.json",
        "software_versions.json",
        "probe_lock.json",
        "policy_state_before.json",
        "policy_state_after.json",
    )
    for name in required:
        path = output / name
        allow_empty = name == "probe_stream.jsonl"
        if not path.is_file() or (path.stat().st_size <= 0 and not allow_empty):
            errors.append(f"missing or empty artifact: {path}")
    episodes: list[dict[str, Any]] = []
    if (output / "episodes.jsonl").is_file():
        try:
            raw_episodes = [
                json.loads(line)
                for line in (output / "episodes.jsonl").read_text().splitlines()
            ]
            if not all(isinstance(row, dict) for row in raw_episodes):
                raise ValueError("episodes.jsonl contains a non-object row")
            episodes = raw_episodes
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid episodes.jsonl: {error}")
    if len(episodes) != expected_count:
        errors.append(f"episode count {len(episodes)} != expected {expected_count}")
    episode_keys = [
        (str(row.get("episode_id")), str(row.get("arm"))) for row in episodes
    ]
    if len(set(episode_keys)) != len(episode_keys):
        errors.append("duplicate episode id/arm pairs")
    if expected_plan:
        expected_keys = {
            (str(item.get("episode_id")), str(item.get("arm"))): (
                int(item.get("seed", -1)),
                str(item.get("condition")),
                str(item.get("pair_id")),
            )
            for item in expected_plan
        }
        observed_keys = [
            (str(row.get("episode_id")), str(row.get("arm"))) for row in episodes
        ]
        if set(observed_keys) != set(expected_keys):
            errors.append("episode ids/arms do not match the frozen episode plan")
        for row in episodes:
            expected = expected_keys.get(
                (str(row.get("episode_id")), str(row.get("arm")))
            )
            if expected is None:
                continue
            expected_seed, expected_condition, expected_pair_id = expected
            if int(row.get("seed", -1)) != expected_seed:
                errors.append(f"seed mismatch in {row.get('episode_id')}")
            if str(row.get("condition")) != expected_condition:
                errors.append(f"condition mismatch in {row.get('episode_id')}")
            if str(row.get("pair_id")) != expected_pair_id:
                errors.append(f"pair id mismatch in {row.get('episode_id')}")
    if stage != "final" and any(int(row.get("seed", -1)) >= 900 for row in episodes):
        errors.append("final-test seed entered a non-final diagnostic run")
    for row in episodes:
        if int(row.get("probe_steps", 0)) > MAX_PROBE_STEPS:
            errors.append(f"probe step budget exceeded in {row.get('episode_id')}")
        if not row.get("monitor_parameter_sha256"):
            errors.append(f"missing monitor hash in {row.get('episode_id')}")
    episode_tokens = {
        str(row.get("episode_token")): row for row in episodes if row.get("episode_token")
    }
    for name in ("monitor_stream.jsonl", "probe_stream.jsonl"):
        path = output / name
        if not path.is_file():
            continue
        try:
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
            ]
            for index, value in enumerate(rows):
                if not isinstance(value, dict):
                    errors.append(f"online stream row is not an object at {name}:{index}")
                    continue
                if name == "monitor_stream.jsonl" and "executed_action" in value:
                    errors.append(f"online monitor leak at {name}:{index}")
                assert_online_event_safe(value)
                token = str(value.get("episode_token", ""))
                if not token:
                    errors.append(f"missing episode token in {name}:{index}")
                elif token not in episode_tokens:
                    errors.append(f"unknown episode token in {name}:{index}")
                if name == "probe_stream.jsonl" and token:
                    arm = str(episode_tokens.get(token, {}).get("arm", ""))
                    if arm == "passive_only":
                        errors.append(f"probe event recorded for passive arm at {name}:{index}")
            if name == "probe_stream.jsonl":
                by_token: dict[str, list[dict[str, Any]]] = {}
                for value in rows:
                    if not isinstance(value, dict):
                        continue
                    by_token.setdefault(str(value.get("episode_token")), []).append(value)
                expected_operations = [
                    "repeat_previous_requested_action",
                    "force_requery_and_execute_first_action",
                    "continue_requeried_chunk",
                    "continue_requeried_chunk",
                ]
                for token, token_rows in by_token.items():
                    ordered = sorted(token_rows, key=lambda row: int(row.get("probe_step", -1)))
                    if [row.get("probe_step") for row in ordered] != list(range(len(ordered))):
                        errors.append(f"probe steps are not consecutive for token {token}")
                    observed_operations = [row.get("operation") for row in ordered]
                    if observed_operations != expected_operations[: len(ordered)]:
                        errors.append(f"probe operations drifted for token {token}")
                    if len(ordered) > MAX_PROBE_STEPS:
                        errors.append(f"probe stream exceeds step budget for token {token}")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"invalid online stream {name}: {error}")
            break
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("debug", "pilot", "final"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_PILOT_CONFIG)
    parser.add_argument("--probe-config", type=Path, default=DEFAULT_PROBE_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--monitor", type=Path, default=DEFAULT_MONITOR)
    parser.add_argument("--monitor-protocol", type=Path, default=DEFAULT_MONITOR_PROTOCOL)
    parser.add_argument("--monitor-metrics", type=Path, default=DEFAULT_MONITOR_METRICS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="target", choices=("target",))
    parser.add_argument("--horizon", type=int, default=750)
    parser.add_argument("--policy-python", default="/home/pc/VLA/envs/groot/bin/python")
    parser.add_argument("--server-script", default=str(ROOT / "scripts/serve_groot_policy.py"))
    parser.add_argument("--port", type=int, default=5588)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--server-start-timeout", type=int, default=900)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("source /home/pc/VLA/env.sh before running the diagnostic probe")
    if args.horizon != 750:
        raise ValueError("the pre-registered diagnostic probe requires horizon=750")
    config = _load_json(args.config)
    probe = _load_json(args.probe_config)
    monitor_metrics = _load_json(args.monitor_metrics)
    protocol_path = args.monitor_protocol
    monitor_protocol = _load_json(protocol_path)
    errors = validate_probe_protocol(probe, monitor_protocol)
    if errors:
        raise ValueError(f"invalid diagnostic probe protocol: {errors}")
    if (
        config["environment"]["id"] != args.environment_id
        or config["environment"]["split"] != args.split
    ):
        raise ValueError("pilot configuration does not match requested environment")
    if (
        monitor_metrics.get("status") != "completed"
        or not monitor_metrics.get("gate", {}).get("passed")
    ):
        raise ValueError("monitor checkpoint is not backed by a passed monitor gate")
    monitor = FaultConditionedTemporalMonitor.load(args.monitor)
    monitor_hash = _sha256(args.monitor)
    risk_threshold = float(monitor.threshold_)
    lock = _validate_lock(
        args.lock,
        monitor_path=args.monitor,
        probe_path=args.probe_config,
        risk_threshold=risk_threshold,
        protocol_path=protocol_path,
    )
    entropy_threshold = float(lock["calibration"]["entropy_threshold"])
    # Do not even materialize final-test seeds on debug/pilot paths.  The
    # separate final lock is checked before that list can be consumed.
    final_seeds = (
        [int(seed) for seed in probe.get("staging", {}).get("final_seeds", [])]
        if args.stage == "final"
        else []
    )
    if args.stage == "final":
        if not lock.get("final_lock", False):
            raise ValueError("final stage requires a separate final-lock artifact")
        if not final_seeds:
            raise ValueError("final stage requires an explicit non-empty final seed list")
    selected_seeds, plan = _plan_for_stage(config, args.stage, final_seeds=final_seeds)
    if args.stage == "final" and not plan:
        raise ValueError("final stage plan is empty; refusing to run")
    if args.stage == "debug" and len(plan) != len(selected_seeds) * 3 * 2:
        raise ValueError("debug stage plan does not contain both arms for all conditions")
    if args.stage == "pilot" and len(plan) != int(probe["staging"]["pilot_episode_count"]):
        raise ValueError("pilot stage episode count does not match the frozen probe protocol")
    expected_count = len(selected_seeds) * 3 * 2
    output = ensure_empty_output_dir(args.output)
    checkpoint_hashes = _verify_checkpoint(_load_json(args.manifest), args.checkpoint)

    write_json_once(output / "probe_lock.json", lock)
    write_json_once(
        output / "monitor_config.json",
        {
            "name": monitor.name,
            "feature_version": "monitor-inputs-v1.0",
            "checkpoint": str(args.monitor.resolve()),
            "checkpoint_sha256": monitor_hash,
            "risk_threshold": risk_threshold,
            "entropy_threshold": entropy_threshold,
            "trigger": "risk_alarm OR normalized_entropy_alarm",
            "allowed_inputs": [
                "previous_observation",
                "current_observation",
                "requested_action",
                "requested_action_chunk",
                "chunk_metadata",
                "task_instruction",
                "declared_policy_latency",
            ],
            "forbidden_inputs": sorted(FORBIDDEN_ONLINE_FIELDS),
            "parameter_hash_before": monitor_parameter_sha256(monitor),
            "parameter_hash_after": monitor_parameter_sha256(monitor),
        },
    )
    write_json_once(
        output / "calibration.json",
        {
            "status": "locked",
            "probe_lock": str(args.lock.resolve()),
            "risk_threshold": risk_threshold,
            "entropy_threshold": entropy_threshold,
        },
    )
    write_json_once(
        output / "software_versions.json",
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
    monitor_parameter_before = monitor_parameter_sha256(monitor)
    results: list[dict[str, Any]] = []
    try:
        process, server_log = _start_server(args, output)
        client = TorchZmqPolicyClient(port=args.port, timeout_ms=300000)
        policy_state_before = client.call("get_policy_state")
        if (
            policy_state_before.get("model_training")
            or not policy_state_before.get("all_parameters_frozen")
        ):
            raise RuntimeError("policy server did not report eval mode with frozen parameters")
        write_json_once(output / "policy_state_before.json", policy_state_before)
        environment = RoboCasaEnvironment(args.environment_id, split=args.split)
        policy = GrootRoboCasaPolicy(environment.action_space, client)
        with (
            (output / "episodes.jsonl").open("x", encoding="utf-8") as episode_stream,
            (output / "monitor_stream.jsonl").open("x", encoding="utf-8") as monitor_stream,
            (output / "probe_stream.jsonl").open("x", encoding="utf-8") as probe_stream,
            (output / "privileged_audit.jsonl").open("x", encoding="utf-8") as audit_stream,
        ):
            for index, item in enumerate(plan):
                result = run_episode(
                    episode_index=index,
                    item=item,
                    arm=str(item["arm"]),
                    environment=environment,
                    policy=policy,
                    client=client,
                    monitor=monitor,
                    risk_threshold=risk_threshold,
                    entropy_threshold=entropy_threshold,
                    base_horizon=args.horizon,
                    monitor_stream=monitor_stream,
                    probe_stream=probe_stream,
                    audit_stream=audit_stream,
                    episode_stream=episode_stream,
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "episode_index",
                                "episode_id",
                                "arm",
                                "condition",
                                "seed",
                                "steps",
                                "triggered",
                                "probe_steps",
                            )
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        policy_state_after = client.call("get_policy_state")
        write_json_once(output / "policy_state_after.json", policy_state_after)
        for label, state in (("before", policy_state_before), ("after", policy_state_after)):
            if state.get("model_training") or not state.get("all_parameters_frozen"):
                raise RuntimeError(f"policy server was not frozen in {label} state")
            if state.get("initial_parameter_sha256") != state.get("current_parameter_sha256"):
                raise RuntimeError(
                    f"policy parameter hash differs from initial hash in {label} state"
                )
        if (
            policy_state_before.get("current_parameter_sha256")
            != policy_state_after.get("current_parameter_sha256")
        ):
            raise RuntimeError("frozen policy parameter hash changed during diagnostic probe")
        monitor_parameter_after = monitor_parameter_sha256(monitor)
        if monitor_parameter_before != monitor_parameter_after:
            raise RuntimeError("monitor parameters changed during diagnostic probe")

        metrics = {
            "status": "completed",
            "scientific_result": False,
            "stage": args.stage,
            "protocol_version": config["protocol_version"],
            "probe_protocol_version": probe["probe_protocol_version"],
            "monitor_protocol": {
                "path": str(protocol_path.resolve()),
                "sha256": _sha256(protocol_path),
                "relock_version": monitor_protocol.get("relock_version"),
            },
            "environment": {
                "id": args.environment_id,
                "split": args.split,
                "horizon": args.horizon,
                "max_probe_steps": MAX_PROBE_STEPS,
            },
            "seeds": selected_seeds,
            "episode_count": len(results),
            "analysis": {
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": 1404,
                "offline_analyzer": "scripts/analyze_diagnostic_probe.py",
            },
            "arms": {
                arm: sum(row["arm"] == arm for row in results)
                for arm in ("passive_only", "passive_plus_probe")
            },
            "triggered_episodes": sum(bool(row["triggered"]) for row in results),
            "probe_invocations": sum(bool(row["probe_executed"]) for row in results),
            "probe_step_maximum": max((int(row["probe_steps"]) for row in results), default=0),
            "clean_joint_trigger_rate": (
                sum(bool(row["triggered"]) for row in results if row["condition"] == "clean")
                / max(1, sum(row["condition"] == "clean" for row in results))
            ),
            "monitor": {
                "checkpoint": str(args.monitor.resolve()),
                "checkpoint_sha256": monitor_hash,
                "parameter_sha256_before": monitor_parameter_before,
                "parameter_sha256_after": monitor_parameter_after,
                "risk_threshold": risk_threshold,
                "entropy_threshold": entropy_threshold,
            },
            "policy": {
                "name": GrootRoboCasaPolicy.name,
                "manifest": str(args.manifest.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_files_sha256": checkpoint_hashes,
                "parameter_sha256_before": policy_state_before.get("current_parameter_sha256"),
                "parameter_sha256_after": policy_state_after.get("current_parameter_sha256"),
                "frozen": True,
            },
            "outputs": {
                "episodes": str(output / "episodes.jsonl"),
                "monitor_stream": str(output / "monitor_stream.jsonl"),
                "probe_stream": str(output / "probe_stream.jsonl"),
                "privileged_audit": str(output / "privileged_audit.jsonl"),
            },
        }
        write_json_once(output / "metrics.json", metrics)
        write_json_once(
            output / "run_manifest.json",
            {
                "status": "completed",
                "scientific_result": False,
                "stage": args.stage,
                "protocol_version": config["protocol_version"],
                "probe_protocol_version": probe["probe_protocol_version"],
                "monitor_protocol": metrics["monitor_protocol"],
                "environment": metrics["environment"],
                "seeds": selected_seeds,
                "episode_plan": plan,
                "probe_lock": {
                    "path": str(args.lock.resolve()),
                    "sha256": _sha256(args.lock),
                },
                "monitor": metrics["monitor"],
                "policy": metrics["policy"],
                "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config)},
                "probe_config": {
                    "path": str(args.probe_config.resolve()),
                    "sha256": _sha256(args.probe_config),
                },
                "analysis": metrics["analysis"],
                "command": [sys.executable, *sys.argv],
            },
        )
        artifact_errors = _validate_run_artifacts(
            output,
            expected_count,
            stage=args.stage,
            expected_plan=plan,
        )
        write_json_once(
            output / "artifact_validation.json",
            {
                "status": "passed" if not artifact_errors else "failed",
                "expected_episode_count": expected_count,
                "errors": artifact_errors,
            },
        )
        if artifact_errors:
            raise RuntimeError(f"diagnostic probe artifact validation failed: {artifact_errors}")
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
