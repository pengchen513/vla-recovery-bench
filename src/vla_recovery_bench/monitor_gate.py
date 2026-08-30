"""Read-only integrity gate for formal monitor dataset shards.

The gate joins privileged labels only while auditing offline artifacts. It does
not load the VLA checkpoint, create an environment, train a model, or modify a
shard. A partition passes only when its declared shards are individually
intact and their disjoint seed union exactly matches the frozen protocol.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .groot_adapter import ACTION_DIMS, ACTION_HORIZON
from .monitor import FEATURE_NAMES, FEATURE_VERSION
from .monitor_dataset import MonitorDatasetEpisode, load_monitor_dataset
from .monitor_protocol import (
    CONDITIONS,
    SPLIT_KEYS,
    monitor_episode_plan,
    validate_monitor_protocol,
    validate_monitor_relock_protocol,
)
from .recording import to_jsonable

GATE_VERSION = "formal-monitor-shard-gate-v1"
INTEGRITY_SCHEMA_VERSION = "monitor-shard-integrity-v1"
FORMAL_PARTITIONS = ("train", "calibration", "validation")
FORMAL_COLLECTION_ROLES = frozenset({"formal_shard", "full_partition"})

HASHED_SHARD_ARTIFACTS = (
    "run_manifest.json",
    "dataset_index.jsonl",
    "monitor_inputs.h5",
    "offline_labels.h5",
    "episodes.jsonl",
    "audit_stream.jsonl",
    "metrics.json",
    "software_versions.json",
    "policy_state_before.json",
    "policy_state_after.json",
    "artifact_validation.json",
)
REQUIRED_SHARD_ARTIFACTS = (*HASHED_SHARD_ARTIFACTS, "shard_integrity.json")
SHARD_ARTIFACT_PATHS = {
    "run_manifest": "run_manifest.json",
    "dataset_index": "dataset_index.jsonl",
    "monitor_inputs": "monitor_inputs.h5",
    "offline_labels": "offline_labels.h5",
    "episodes": "episodes.jsonl",
    "audit_stream": "audit_stream.jsonl",
    "metrics": "metrics.json",
    "software_versions": "software_versions.json",
    "policy_state_before": "policy_state_before.json",
    "policy_state_after": "policy_state_after.json",
    "artifact_validation": "artifact_validation.json",
    "shard_integrity": "shard_integrity.json",
}
ACTION_FEATURE_START = FEATURE_NAMES.index("requested_action_chunk_0")
ACTION_FEATURE_COUNT = ACTION_HORIZON * sum(ACTION_DIMS.values())
CHUNK_POSITION_INDEX = FEATURE_NAMES.index("chunk_position")
CHUNK_LENGTH_INDEX = FEATURE_NAMES.index("chunk_length")
REMAINING_HORIZON_INDEX = FEATURE_NAMES.index("remaining_horizon")
POLICY_LATENCY_INDEX = FEATURE_NAMES.index("policy_latency_ms")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_token(protocol_sha256: str, episode_id: str) -> str:
    return hashlib.sha256(
        f"monitor-input-channel-v1|{protocol_sha256}|{episode_id}".encode()
    ).hexdigest()[:32]


def build_shard_integrity_manifest(
    output_dir: str | Path,
    *,
    partition: str,
    collection_role: str,
    seeds: Sequence[int],
    protocol_sha256: str,
    policy_manifest_sha256: str,
) -> dict[str, Any]:
    """Build, but do not write, the write-once checksum manifest for a shard."""
    output = Path(output_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    for name in HASHED_SHARD_ARTIFACTS:
        path = output / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"cannot seal missing or empty shard artifact: {path}")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "status": "sealed",
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "partition": partition,
        "collection_role": collection_role,
        "seeds": [int(seed) for seed in seeds],
        "protocol_sha256": protocol_sha256,
        "policy_manifest_sha256": policy_manifest_sha256,
        "artifacts": artifacts,
    }


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_reject_nonfinite_json)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _expected_policy_files(policy_manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(record["path"]): str(record["sha256"])
        for record in policy_manifest.get("checkpoint_files", [])
    }


def _same(actual: Any, expected: Any, field: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{field} mismatch: expected={expected!r}, got={actual!r}")


def _validate_action(
    action: Any,
    action_space: Mapping[str, Any],
    *,
    context: str,
) -> list[str]:
    errors: list[str] = []
    spaces = action_space.get("spaces")
    if not isinstance(spaces, Mapping):
        return ["environment action_space.spaces is missing"]
    if not isinstance(action, Mapping):
        return [f"{context} is not a structured action mapping"]
    expected_keys = set(spaces)
    actual_keys = set(action)
    if actual_keys != expected_keys:
        return [
            f"{context} action keys mismatch: missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        ]
    for key, contract in spaces.items():
        if not isinstance(contract, Mapping):
            errors.append(f"action contract for {key} is not an object")
            continue
        try:
            values = np.asarray(action[key], dtype=np.float64)
            low = np.asarray(contract["low"], dtype=np.float64)
            high = np.asarray(contract["high"], dtype=np.float64)
            shape = tuple(int(value) for value in contract["shape"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"invalid {context}.{key} or action contract: {error}")
            continue
        if values.shape != shape or low.shape != shape or high.shape != shape:
            errors.append(
                f"{context}.{key} shape mismatch: expected={shape}, got={values.shape}"
            )
            continue
        if not np.all(np.isfinite(values)):
            errors.append(f"{context}.{key} contains NaN or Inf")
            continue
        if np.any(values < low - 1e-6) or np.any(values > high + 1e-6):
            errors.append(
                f"{context}.{key} exceeds action bounds: "
                f"range=[{float(values.min())}, {float(values.max())}]"
            )
    return errors


def _expected_exposure(episode: MonitorDatasetEpisode) -> np.ndarray:
    if episode.condition == "clean":
        return np.zeros(len(episode.features), dtype=np.bool_)
    if not isinstance(episode.factor_row, Mapping):
        raise ValueError("factor_row is missing")
    onset = int(episode.factor_row["onset_step"])
    end = onset + int(episode.factor_row["duration_steps"])
    reference = (
        episode.observation_steps
        if episode.condition == "observation_fault"
        else episode.control_steps
    )
    return np.logical_and(reference >= onset, reference < end)


def _flatten_action(action: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(action[key], dtype=np.float64).reshape(-1) for key in ACTION_DIMS]
    )


def _validate_dataset_episodes(
    episodes: list[MonitorDatasetEpisode],
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    partition: str,
    seeds: list[int],
) -> tuple[list[str], dict[str, MonitorDatasetEpisode]]:
    errors: list[str] = []
    expected_plan = monitor_episode_plan(protocol, partition, seeds=seeds)
    expected_by_key = {
        (int(item["seed"]), str(item["condition"])): item for item in expected_plan
    }
    observed_by_key: dict[tuple[int, str], MonitorDatasetEpisode] = {}
    by_token: dict[str, MonitorDatasetEpisode] = {}
    for episode in episodes:
        key = (episode.seed, episode.condition)
        if key in observed_by_key:
            errors.append(f"duplicate HDF5 seed/condition pair: {key}")
            continue
        observed_by_key[key] = episode
        if episode.token in by_token:
            errors.append(f"duplicate HDF5 episode token: {episode.token}")
        by_token[episode.token] = episode
        expected = expected_by_key.get(key)
        if expected is None:
            errors.append(f"unexpected HDF5 seed/condition pair: {key}")
            continue
        expected_mechanism = str(expected["mechanism"])
        expected_schedule = to_jsonable(expected["faults"])
        _same(episode.partition, partition, f"episode {episode.token} partition", errors)
        _same(
            episode.pair_id,
            str(expected["pair_id"]),
            f"episode {episode.token} pair_id",
            errors,
        )
        _same(
            episode.episode_id,
            str(expected["episode_id"]),
            f"episode {episode.token} episode_id",
            errors,
        )
        _same(
            episode.mechanism,
            expected_mechanism,
            f"episode {episode.token} mechanism",
            errors,
        )
        _same(
            episode.factor_row,
            to_jsonable(expected["factor_row"]),
            f"episode {episode.token} factor_row",
            errors,
        )
        _same(
            episode.fault_schedule,
            expected_schedule,
            f"episode {episode.token} fault_schedule",
            errors,
        )
        expected_token = episode_token(protocol_sha256, str(expected["episode_id"]))
        _same(episode.token, expected_token, f"episode {key} token", errors)
        rows = len(episode.features)
        if rows <= 0:
            errors.append(f"episode {episode.token} has no feature rows")
        if episode.features.ndim != 2 or episode.features.shape[1] != len(FEATURE_NAMES):
            errors.append(f"episode {episode.token} has invalid feature shape")
        if not np.all(np.isfinite(episode.features)):
            errors.append(f"episode {episode.token} contains non-finite features")
        action_chunk = episode.features[
            :, ACTION_FEATURE_START : ACTION_FEATURE_START + ACTION_FEATURE_COUNT
        ]
        if np.any(action_chunk < -1.0 - 1e-6) or np.any(action_chunk > 1.0 + 1e-6):
            errors.append(f"episode {episode.token} requested action chunk exceeds [-1, 1]")
        if episode.control_steps.shape != (rows,) or episode.observation_steps.shape != (rows,):
            errors.append(f"episode {episode.token} step arrays do not match feature rows")
        elif not np.array_equal(episode.control_steps, np.arange(rows, dtype=np.int32)):
            errors.append(f"episode {episode.token} control steps are not contiguous from zero")
        elif not np.array_equal(episode.observation_steps, episode.control_steps + 1):
            errors.append(f"episode {episode.token} observation/control timestamps disagree")
        if rows:
            expected_positions = episode.control_steps % ACTION_HORIZON
            if not np.array_equal(
                episode.features[:, CHUNK_POSITION_INDEX], expected_positions
            ):
                errors.append(f"episode {episode.token} chunk positions are inconsistent")
            if not np.all(episode.features[:, CHUNK_LENGTH_INDEX] == ACTION_HORIZON):
                errors.append(f"episode {episode.token} chunk length is not {ACTION_HORIZON}")
            remaining = int(protocol["environment"]["horizon"]) - episode.control_steps - 1
            if not np.array_equal(episode.features[:, REMAINING_HORIZON_INDEX], remaining):
                errors.append(f"episode {episode.token} remaining horizon is inconsistent")
            if np.any(episode.features[:, POLICY_LATENCY_INDEX] < 0):
                errors.append(f"episode {episode.token} has negative policy latency")
        if episode.exposure.shape != (rows,):
            errors.append(f"episode {episode.token} exposure rows do not match features")
        else:
            try:
                expected_exposure = _expected_exposure(episode)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"episode {episode.token} has invalid exposure metadata: {error}")
            else:
                if not np.array_equal(episode.exposure, expected_exposure):
                    errors.append(f"episode {episode.token} exposure mask is not reproducible")
        if not episode.instruction:
            errors.append(f"episode {episode.token} has an empty task instruction")
    missing = sorted(set(expected_by_key) - set(observed_by_key))
    if missing:
        errors.append(f"missing HDF5 seed/condition pairs: {missing}")
    return errors, by_token


def _validate_episode_jsonl(
    directory: Path,
    *,
    by_token: Mapping[str, MonitorDatasetEpisode],
) -> list[str]:
    errors: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    indices: dict[str, dict[str, Any]] = {}
    try:
        for _, record in _iter_jsonl(directory / "episodes.jsonl"):
            token = str(record.get("episode_token", ""))
            if record.get("event_type") != "episode":
                errors.append(f"episodes.jsonl has unexpected event_type for {token}")
            if not token or token in records:
                errors.append(f"episodes.jsonl has missing or duplicate token: {token!r}")
            records[token] = record
        for _, record in _iter_jsonl(directory / "dataset_index.jsonl"):
            token = str(record.get("episode_token", ""))
            if record.get("event_type") != "dataset_episode":
                errors.append(f"dataset_index.jsonl has unexpected event_type for {token}")
            if not token or token in indices:
                errors.append(f"dataset_index.jsonl has missing or duplicate token: {token!r}")
            indices[token] = record
    except (OSError, ValueError) as error:
        return [str(error)]
    expected_tokens = set(by_token)
    for name, values in (("episodes.jsonl", records), ("dataset_index.jsonl", indices)):
        if set(values) != expected_tokens:
            errors.append(
                f"{name} token set disagrees with HDF5: "
                f"missing={sorted(expected_tokens - set(values))}, "
                f"extra={sorted(set(values) - expected_tokens)}"
            )
    for token in sorted(expected_tokens & set(records) & set(indices)):
        episode = by_token[token]
        summary = records[token]
        index = indices[token]
        expected_fields = {
            "episode_token": token,
            "episode_id": episode.episode_id,
            "pair_id": episode.pair_id,
            "partition": episode.partition,
            "seed": episode.seed,
            "feature_rows": len(episode.features),
            "input_group": f"/episodes/{token}",
            "label_group": f"/episodes/{token}",
        }
        for field, expected in expected_fields.items():
            _same(index.get(field), expected, f"index {token}.{field}", errors)
            _same(summary.get(field), expected, f"episode {token}.{field}", errors)
        _same(summary.get("condition"), episode.condition, f"episode {token}.condition", errors)
        _same(summary.get("mechanism"), episode.mechanism, f"episode {token}.mechanism", errors)
        _same(summary.get("steps"), len(episode.features), f"episode {token}.steps", errors)
        _same(
            summary.get("exposed_rows"),
            int(episode.exposure.sum()),
            f"episode {token}.exposed_rows",
            errors,
        )
        _same(
            summary.get("not_exposed"),
            bool(episode.condition != "clean" and not episode.exposure.any()),
            f"episode {token}.not_exposed",
            errors,
        )
        expected_applied_faults = sum(
            int(fault["step"]) < len(episode.features)
            for fault in (episode.fault_schedule or [])
        )
        _same(
            summary.get("configured_fault_count"),
            len(episode.fault_schedule or []),
            f"episode {token}.configured_fault_count",
            errors,
        )
        _same(
            summary.get("applied_fault_count"),
            expected_applied_faults,
            f"episode {token}.applied_fault_count",
            errors,
        )
        _same(summary.get("success"), episode.success, f"episode {token}.success", errors)
        saturated = summary.get("action_saturated_values")
        if (
            not isinstance(saturated, int)
            or isinstance(saturated, bool)
            or saturated < 0
        ):
            errors.append(f"episode {token}.action_saturated_values is not non-negative")
    return errors


def _validate_audit_jsonl(
    path: Path,
    *,
    by_token: Mapping[str, MonitorDatasetEpisode],
    action_space: Mapping[str, Any],
) -> tuple[list[str], int]:
    errors: list[str] = []
    transition_counts: Counter[str] = Counter()
    transition_steps: dict[str, list[int]] = {token: [] for token in by_token}
    fault_records: dict[str, list[dict[str, Any]]] = {token: [] for token in by_token}
    try:
        for line_number, record in _iter_jsonl(path):
            token = str(record.get("episode_token", ""))
            if token not in by_token:
                errors.append(f"audit line {line_number} references unknown token {token!r}")
                continue
            event_type = record.get("event_type")
            if event_type == "fault_injection":
                fault_records[token].append(record)
                continue
            if event_type != "audit_transition":
                errors.append(f"audit line {line_number} has unknown event_type {event_type!r}")
                continue
            transition_counts[token] += 1
            episode = by_token[token]
            try:
                control_step = int(record["control_step"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"audit line {line_number} has invalid control_step")
                continue
            transition_steps[token].append(control_step)
            for field, expected in (
                ("episode_id", episode.episode_id),
                ("pair_id", episode.pair_id),
                ("condition", episode.condition),
                ("seed", episode.seed),
            ):
                _same(
                    record.get(field),
                    expected,
                    f"audit line {line_number}.{field}",
                    errors,
                )
            audit = record.get("audit")
            if not isinstance(audit, Mapping):
                errors.append(f"audit line {line_number} has no audit object")
                continue
            errors.extend(
                _validate_action(
                    audit.get("requested_action"),
                    action_space,
                    context=f"audit line {line_number} requested_action",
                )
            )
            if audit.get("step") != control_step:
                errors.append(f"audit line {line_number} embedded step disagrees")
            requested = audit.get("requested_action")
            if isinstance(requested, Mapping) and 0 <= control_step < len(episode.features):
                try:
                    flat_action = _flatten_action(requested)
                    chunk_position = int(
                        episode.features[control_step, CHUNK_POSITION_INDEX]
                    )
                    start = ACTION_FEATURE_START + chunk_position * flat_action.size
                    declared = episode.features[
                        control_step, start : start + flat_action.size
                    ]
                    if declared.shape != flat_action.shape or not np.allclose(
                        declared, flat_action, rtol=0.0, atol=1e-6
                    ):
                        errors.append(
                            f"audit line {line_number} requested action disagrees "
                            "with its declared action-chunk position"
                        )
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"audit line {line_number} action-chunk join failed: {error}")
            errors.extend(
                _validate_action(
                    audit.get("executed_action"),
                    action_space,
                    context=f"audit line {line_number} executed_action",
                )
            )
    except (OSError, ValueError) as error:
        return [str(error)], 0
    for token, episode in by_token.items():
        expected_transitions = len(episode.features)
        if transition_counts[token] != expected_transitions:
            errors.append(
                f"audit transitions for {token}: expected={expected_transitions}, "
                f"got={transition_counts[token]}"
            )
        if transition_steps[token] != list(range(expected_transitions)):
            errors.append(f"audit control steps for {token} are not contiguous from zero")
        expected_faults = [
            fault
            for fault in (episode.fault_schedule or [])
            if int(fault["step"]) < expected_transitions
        ]
        if len(fault_records[token]) != len(expected_faults):
            errors.append(
                f"fault audit count for {token}: expected={len(expected_faults)}, "
                f"got={len(fault_records[token])}"
            )
        for fault_index, (record, expected_fault) in enumerate(
            zip(fault_records[token], expected_faults, strict=False)
        ):
            _same(
                record.get("fault"),
                expected_fault,
                f"fault audit {token}[{fault_index}].fault",
                errors,
            )
            _same(
                record.get("control_step"),
                expected_fault["step"],
                f"fault audit {token}[{fault_index}].control_step",
                errors,
            )
            for field, expected in (
                ("episode_id", episode.episode_id),
                ("condition", episode.condition),
                ("seed", episode.seed),
                ("phase", expected_fault["phase"]),
            ):
                _same(
                    record.get(field),
                    expected,
                    f"fault audit {token}[{fault_index}].{field}",
                    errors,
                )
            application = record.get("application")
            if not isinstance(application, Mapping):
                errors.append(
                    f"fault audit {token}[{fault_index}] has no application object"
                )
                continue
            for field, expected in (
                ("fault_id", expected_fault["fault_id"]),
                ("kind", expected_fault["kind"]),
                ("requested_step", expected_fault["step"]),
                ("applied", True),
            ):
                _same(
                    application.get(field),
                    expected,
                    f"fault audit {token}[{fault_index}].application.{field}",
                    errors,
                )
    return errors, sum(transition_counts.values())


def _validate_shard(
    directory: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    policy_manifest: Mapping[str, Any],
    policy_manifest_sha256: str,
    partition: str,
    expected_seeds: list[int],
) -> dict[str, Any]:
    errors: list[str] = []
    report: dict[str, Any] = {
        "path": str(directory),
        "status": "blocked",
        "seeds": [],
        "episode_count": 0,
        "rows": 0,
        "audit_transition_count": 0,
        "errors": errors,
    }
    if not directory.is_dir():
        errors.append(f"shard directory does not exist: {directory}")
        return report
    missing = [
        name
        for name in REQUIRED_SHARD_ARTIFACTS
        if not (directory / name).is_file() or (directory / name).stat().st_size <= 0
    ]
    if missing:
        errors.append(f"missing or empty required artifacts: {missing}")
    if any(name != "shard_integrity.json" for name in missing):
        return report
    try:
        metrics = _load_json(directory / "metrics.json")
        manifest = _load_json(directory / "run_manifest.json")
        validation = _load_json(directory / "artifact_validation.json")
        integrity = (
            _load_json(directory / "shard_integrity.json")
            if "shard_integrity.json" not in missing
            else {}
        )
        software = _load_json(directory / "software_versions.json")
        before = _load_json(directory / "policy_state_before.json")
        after = _load_json(directory / "policy_state_after.json")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid JSON artifact: {error}")
        return report

    if validation.get("status") != "passed" or validation.get("errors") not in ([], None):
        errors.append("artifact_validation.json did not pass cleanly")
    if integrity.get("schema_version") != INTEGRITY_SCHEMA_VERSION:
        errors.append("shard integrity schema version mismatch")
    if integrity.get("status") != "sealed":
        errors.append("shard integrity manifest is not sealed")

    for source_name, source in (("metrics", metrics), ("manifest", manifest)):
        _same(source.get("status"), "completed", f"{source_name}.status", errors)
        _same(source.get("debug"), False, f"{source_name}.debug", errors)
        _same(
            source.get("scientific_result"),
            False,
            f"{source_name}.scientific_result",
            errors,
        )
        _same(source.get("partition"), partition, f"{source_name}.partition", errors)
        _same(
            source.get("protocol_version"),
            protocol.get("protocol_version"),
            f"{source_name}.protocol_version",
            errors,
        )
        _same(
            source.get("monitor_protocol_version"),
            protocol.get("monitor_protocol_version"),
            f"{source_name}.monitor_protocol_version",
            errors,
        )

    role = metrics.get("collection_role")
    report["collection_role"] = role
    _same(manifest.get("collection_role"), role, "manifest.collection_role", errors)
    _same(integrity.get("collection_role"), role, "integrity.collection_role", errors)
    if role not in FORMAL_COLLECTION_ROLES:
        errors.append(f"collection_role is not formal: {role!r}")
    try:
        seeds = [int(seed) for seed in metrics.get("seeds", [])]
    except (TypeError, ValueError):
        seeds = []
        errors.append("metrics.seeds is not a list of integers")
    report["seeds"] = seeds
    if not seeds or seeds != sorted(seeds) or len(seeds) != len(set(seeds)):
        errors.append("shard seeds must be non-empty, sorted, and unique")
    _same(manifest.get("seeds"), seeds, "manifest.seeds", errors)
    _same(integrity.get("seeds"), seeds, "integrity.seeds", errors)
    extra_seeds = sorted(set(seeds) - set(expected_seeds))
    if extra_seeds:
        errors.append(f"shard contains seeds outside {partition}: {extra_seeds}")
    complete = seeds == expected_seeds
    expected_complete_flag = role == "full_partition"
    _same(
        metrics.get("partition_complete"),
        expected_complete_flag,
        "metrics.partition_complete",
        errors,
    )
    _same(
        manifest.get("partition_complete"),
        expected_complete_flag,
        "manifest.partition_complete",
        errors,
    )
    if role == "formal_shard" and complete:
        errors.append("formal_shard must be a strict partition subset")
    if role == "full_partition" and not complete:
        errors.append("full_partition does not contain the full declared seed list")

    _same(integrity.get("partition"), partition, "integrity.partition", errors)
    _same(
        integrity.get("protocol_sha256"),
        protocol_sha256,
        "integrity.protocol_sha256",
        errors,
    )
    _same(
        integrity.get("policy_manifest_sha256"),
        policy_manifest_sha256,
        "integrity.policy_manifest_sha256",
        errors,
    )
    _same(
        manifest.get("config", {}).get("sha256"),
        protocol_sha256,
        "manifest.config.sha256",
        errors,
    )
    _same(
        manifest.get("policy_manifest", {}).get("sha256"),
        policy_manifest_sha256,
        "manifest.policy_manifest.sha256",
        errors,
    )
    _same(
        manifest.get("monitor_inputs"),
        protocol.get("information_boundary"),
        "manifest.monitor_inputs",
        errors,
    )
    _same(manifest.get("storage"), protocol.get("storage"), "manifest.storage", errors)
    outputs = metrics.get("outputs")
    artifacts = manifest.get("artifacts")
    if not isinstance(outputs, Mapping) or not isinstance(artifacts, Mapping):
        errors.append("metrics.outputs and manifest.artifacts must be objects")
    else:
        _same(artifacts, outputs, "manifest.artifacts", errors)
        missing_keys = sorted(set(SHARD_ARTIFACT_PATHS) - set(outputs))
        extra_keys = sorted(set(outputs) - set(SHARD_ARTIFACT_PATHS))
        if missing_keys or extra_keys:
            errors.append(
                f"artifact path set mismatch: missing={missing_keys}, extra={extra_keys}"
            )
        for key, filename in SHARD_ARTIFACT_PATHS.items():
            if key in outputs and Path(str(outputs[key])).name != filename:
                errors.append(
                    f"artifact path {key} must name {filename}, got={outputs[key]!r}"
                )

    expected_environment = protocol["environment"]
    for source_name, source in (("metrics", metrics), ("manifest", manifest)):
        environment = source.get("environment", {})
        _same(
            environment.get("id"),
            expected_environment.get("id"),
            f"{source_name}.environment.id",
            errors,
        )
        _same(
            environment.get("split"),
            expected_environment.get("split"),
            f"{source_name}.environment.split",
            errors,
        )
        _same(
            environment.get("horizon"),
            expected_environment.get("horizon"),
            f"{source_name}.environment.horizon",
            errors,
        )
    _same(
        manifest.get("environment"),
        metrics.get("environment"),
        "manifest.environment",
        errors,
    )
    _same(
        metrics.get("environment", {}).get("action_space"),
        policy_manifest.get("action_space"),
        "environment/policy manifest action_space",
        errors,
    )

    expected_policy = {
        "name": str(policy_manifest.get("policy_name")),
        "checkpoint_sha256": str(policy_manifest.get("checkpoint_sha256")),
        "checkpoint_files_sha256": _expected_policy_files(policy_manifest),
    }
    for source_name, source in (("metrics", metrics), ("manifest", manifest)):
        policy = source.get("policy", {})
        for field, expected in expected_policy.items():
            _same(policy.get(field), expected, f"{source_name}.policy.{field}", errors)
        _same(policy.get("frozen"), True, f"{source_name}.policy.frozen", errors)
        before_hash = policy.get("parameter_sha256_before")
        after_hash = policy.get("parameter_sha256_after")
        if not before_hash or before_hash != after_hash:
            errors.append(f"{source_name} policy parameter hashes are missing or changed")
    _same(manifest.get("policy"), metrics.get("policy"), "manifest.policy", errors)
    before_hash = before.get("current_parameter_sha256")
    after_hash = after.get("current_parameter_sha256")
    if not before_hash or before_hash != after_hash:
        errors.append("policy_state parameter hash changed or is missing")
    for state_name, state, current_hash in (
        ("policy_state_before", before, before_hash),
        ("policy_state_after", after, after_hash),
    ):
        initial_hash = state.get("initial_parameter_sha256")
        if not initial_hash or initial_hash != current_hash:
            errors.append(
                f"{state_name} initial/current parameter hashes are missing or changed"
            )
    _same(
        metrics.get("policy", {}).get("parameter_sha256_before"),
        before_hash,
        "metrics policy/before state hash",
        errors,
    )
    _same(
        metrics.get("policy", {}).get("parameter_sha256_after"),
        after_hash,
        "metrics policy/after state hash",
        errors,
    )
    if before.get("model_training") or after.get("model_training"):
        errors.append("frozen policy entered training mode")
    if before.get("all_parameters_frozen") is not True:
        errors.append("policy_state_before does not report all parameters frozen")
    if after.get("all_parameters_frozen") is not True:
        errors.append("policy_state_after does not report all parameters frozen")

    required_commits = ("repository_commit", "robocasa_commit", "robosuite_commit", "groot_commit")
    for field in required_commits:
        if not software.get(field):
            errors.append(f"software_versions.{field} is missing")
    if software.get("repository_dirty") is not False:
        errors.append("formal shard requires a clean repository snapshot")
    if not isinstance(software.get("packages"), Mapping):
        errors.append("software_versions.packages is missing")

    declared_hashes = integrity.get("artifacts")
    if not isinstance(declared_hashes, Mapping):
        errors.append("shard_integrity.artifacts is missing")
    else:
        missing_hashes = sorted(set(HASHED_SHARD_ARTIFACTS) - set(declared_hashes))
        extra_hashes = sorted(set(declared_hashes) - set(HASHED_SHARD_ARTIFACTS))
        if missing_hashes or extra_hashes:
            errors.append(
                f"integrity artifact set mismatch: missing={missing_hashes}, extra={extra_hashes}"
            )
        for name in HASHED_SHARD_ARTIFACTS:
            record = declared_hashes.get(name)
            if not isinstance(record, Mapping):
                continue
            path = directory / name
            _same(record.get("bytes"), path.stat().st_size, f"integrity {name}.bytes", errors)
            _same(record.get("sha256"), sha256_file(path), f"integrity {name}.sha256", errors)

    try:
        import h5py

        with (
            h5py.File(directory / "monitor_inputs.h5", "r") as inputs,
            h5py.File(directory / "offline_labels.h5", "r") as labels,
        ):
            input_protocol = str(inputs.attrs.get("protocol_sha256", ""))
            label_protocol = str(labels.attrs.get("protocol_sha256", ""))
            _same(input_protocol, protocol_sha256, "monitor_inputs protocol hash", errors)
            _same(label_protocol, protocol_sha256, "offline_labels protocol hash", errors)
            _same(
                str(inputs.attrs.get("feature_version", "")),
                FEATURE_VERSION,
                "monitor_inputs feature version",
                errors,
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid HDF5 root contract: {error}")

    try:
        episodes = load_monitor_dataset(directory, expected_partition=partition)
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid monitor dataset: {error}")
        episodes = []
    try:
        dataset_errors, by_token = _validate_dataset_episodes(
            episodes,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            partition=partition,
            seeds=seeds,
        )
    except (KeyError, TypeError, ValueError) as error:
        dataset_errors = [f"cannot construct expected episode plan: {error}"]
        by_token = {}
    errors.extend(dataset_errors)
    report["episode_count"] = len(episodes)
    report["rows"] = sum(len(episode.features) for episode in episodes)
    expected_episode_count = len(seeds) * len(CONDITIONS)
    _same(
        metrics.get("episode_count"),
        expected_episode_count,
        "metrics.episode_count",
        errors,
    )
    _same(len(episodes), expected_episode_count, "HDF5 episode count", errors)
    _same(metrics.get("rows"), report["rows"], "metrics.rows", errors)
    _same(metrics.get("feature_version"), FEATURE_VERSION, "metrics.feature_version", errors)
    _same(metrics.get("feature_count"), len(FEATURE_NAMES), "metrics.feature_count", errors)
    _same(
        metrics.get("conditions"),
        {condition: len(seeds) for condition in CONDITIONS},
        "metrics.conditions",
        errors,
    )
    _same(
        metrics.get("exposed_fault_episodes"),
        sum(
            episode.condition != "clean" and bool(episode.exposure.any())
            for episode in episodes
        ),
        "metrics.exposed_fault_episodes",
        errors,
    )
    _same(
        metrics.get("not_exposed_fault_episodes"),
        sum(
            episode.condition != "clean" and not bool(episode.exposure.any())
            for episode in episodes
        ),
        "metrics.not_exposed_fault_episodes",
        errors,
    )
    try:
        expected_plan = to_jsonable(monitor_episode_plan(protocol, partition, seeds=seeds))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"cannot construct manifest episode plan: {error}")
    else:
        _same(manifest.get("episode_plan"), expected_plan, "manifest.episode_plan", errors)
    errors.extend(_validate_episode_jsonl(directory, by_token=by_token))
    audit_errors, audit_count = _validate_audit_jsonl(
        directory / "audit_stream.jsonl",
        by_token=by_token,
        action_space=metrics.get("environment", {}).get("action_space", {}),
    )
    errors.extend(audit_errors)
    report["audit_transition_count"] = audit_count
    _same(audit_count, report["rows"], "audit transition/feature row count", errors)

    provenance = {
        "protocol_sha256": protocol_sha256,
        "policy_manifest_sha256": policy_manifest_sha256,
        "environment": metrics.get("environment"),
        "policy": metrics.get("policy"),
        "software_versions": software,
    }
    report["provenance_fingerprint"] = _canonical_sha256(provenance)
    report["status"] = "passed" if not errors else "blocked"
    return report


def validate_formal_shard_set(
    protocol_path: str | Path,
    policy_manifest_path: str | Path,
    *,
    partition: str,
    shard_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Validate exact, disjoint coverage of one formal protocol partition."""
    protocol_file = Path(protocol_path).resolve()
    policy_file = Path(policy_manifest_path).resolve()
    errors: list[str] = []
    report: dict[str, Any] = {
        "gate_version": GATE_VERSION,
        "status": "blocked",
        "passed": False,
        "partition": partition,
        "protocol": {"path": str(protocol_file)},
        "policy_manifest": {"path": str(policy_file)},
        "shard_count": len(shard_paths),
        "expected_seed_count": 0,
        "observed_seed_count": 0,
        "missing_seeds": [],
        "extra_seeds": [],
        "duplicate_seeds": [],
        "expected_episode_count": 0,
        "observed_episode_count": 0,
        "shards": [],
        "errors": errors,
    }
    if partition not in FORMAL_PARTITIONS:
        errors.append(f"formal shard gate does not admit partition {partition!r}")
        return report
    if not shard_paths:
        errors.append("at least one formal shard path is required")
        return report
    resolved = [Path(path).resolve() for path in shard_paths]
    duplicate_paths = sorted(str(path) for path, count in Counter(resolved).items() if count > 1)
    if duplicate_paths:
        errors.append(f"duplicate shard directories: {duplicate_paths}")
    try:
        protocol = _load_json(protocol_file)
        policy_manifest = _load_json(policy_file)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"cannot load gate configuration: {error}")
        return report
    protocol_errors = validate_monitor_protocol(protocol)
    if protocol_errors:
        errors.extend(f"invalid monitor protocol: {error}" for error in protocol_errors)
        return report
    protocol_sha256 = sha256_file(protocol_file)
    policy_manifest_sha256 = sha256_file(policy_file)
    report["protocol"]["sha256"] = protocol_sha256
    report["policy_manifest"]["sha256"] = policy_manifest_sha256
    if protocol.get("relock_version") is not None:
        parent_reference = Path(str(protocol.get("parent_monitor_protocol", "")))
        parent_candidates = [
            protocol_file.parent / parent_reference,
            protocol_file.parents[1] / parent_reference,
            parent_reference,
        ]
        parent_file = next(
            (candidate.resolve() for candidate in parent_candidates if candidate.is_file()),
            None,
        )
        if parent_file is None:
            errors.append(
                "monitor relock parent protocol is missing: "
                f"{protocol.get('parent_monitor_protocol')}"
            )
            return report
        try:
            parent_config = _load_json(parent_file)
            parent_sha256 = sha256_file(parent_file)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"cannot load monitor relock parent protocol: {error}")
            return report
        relock_errors = validate_monitor_relock_protocol(
            protocol, parent_config=parent_config, parent_sha256=parent_sha256
        )
        if relock_errors:
            errors.extend(f"invalid monitor relock: {error}" for error in relock_errors)
            return report
        report["protocol"]["relock_version"] = protocol.get("relock_version")
        report["protocol"]["parent_path"] = str(parent_file)
        report["protocol"]["parent_sha256"] = parent_sha256
    policy_contract_invalid = False
    if protocol.get("policy", {}).get("name") != policy_manifest.get("policy_name"):
        errors.append("monitor protocol and policy manifest names disagree")
        policy_contract_invalid = True
    if protocol.get("policy", {}).get("checkpoint_sha256") != policy_manifest.get(
        "checkpoint_sha256"
    ):
        errors.append("monitor protocol and policy manifest checkpoint hashes disagree")
        policy_contract_invalid = True
    if policy_contract_invalid:
        return report
    expected_seeds = [int(seed) for seed in protocol["splits"][SPLIT_KEYS[partition]]]
    report["expected_seed_count"] = len(expected_seeds)
    report["expected_episode_count"] = len(expected_seeds) * len(CONDITIONS)

    shard_reports = [
        _validate_shard(
            directory,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            policy_manifest=policy_manifest,
            policy_manifest_sha256=policy_manifest_sha256,
            partition=partition,
            expected_seeds=expected_seeds,
        )
        for directory in resolved
    ]
    report["shards"] = shard_reports
    for shard in shard_reports:
        errors.extend(f"{shard['path']}: {error}" for error in shard["errors"])

    observed_seeds = [seed for shard in shard_reports for seed in shard["seeds"]]
    seed_counts = Counter(observed_seeds)
    observed_set = set(observed_seeds)
    expected_set = set(expected_seeds)
    report["observed_seed_count"] = len(observed_set)
    report["missing_seeds"] = sorted(expected_set - observed_set)
    report["extra_seeds"] = sorted(observed_set - expected_set)
    report["duplicate_seeds"] = sorted(seed for seed, count in seed_counts.items() if count > 1)
    report["observed_episode_count"] = sum(
        int(shard["episode_count"]) for shard in shard_reports
    )
    if report["missing_seeds"]:
        errors.append(f"partition is missing seeds: {report['missing_seeds']}")
    if report["extra_seeds"]:
        errors.append(f"partition has out-of-split seeds: {report['extra_seeds']}")
    if report["duplicate_seeds"]:
        errors.append(f"seeds occur in multiple shards: {report['duplicate_seeds']}")
    if report["observed_episode_count"] != report["expected_episode_count"]:
        errors.append(
            "partition episode count mismatch: "
            f"expected={report['expected_episode_count']}, "
            f"got={report['observed_episode_count']}"
        )
    fingerprints = {
        shard.get("provenance_fingerprint")
        for shard in shard_reports
        if shard.get("provenance_fingerprint")
    }
    if len(fingerprints) != 1:
        errors.append("formal shards do not share one identical provenance fingerprint")
    if len(shard_reports) > 1 and any(
        shard.get("collection_role") == "full_partition" for shard in shard_reports
    ):
        errors.append("a full_partition artifact cannot be combined with other shards")
    report["provenance_fingerprint"] = next(iter(fingerprints)) if len(fingerprints) == 1 else None
    report["status"] = "passed" if not errors else "blocked"
    report["passed"] = not errors
    return report


def _resolve_protocol_reference(reference: str | Path, *, relative_to: Path) -> Path:
    """Resolve a repository-relative protocol reference without changing files."""
    candidate = Path(reference)
    candidates = (
        candidate,
        relative_to / candidate,
        relative_to.parent / candidate,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return (relative_to / candidate).resolve()


def _protocol_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must remain identical across monitor data sources."""
    # Relock metadata and scene-seed assignments are intentionally allowed to
    # differ.  The remaining fields define the observation, fault, model, and
    # storage contract that makes rows comparable across source protocols.
    fields = (
        "protocol_version",
        "monitor_protocol_version",
        "environment",
        "policy",
        "information_boundary",
        "fault_sampling",
        "collection_design",
        "storage",
        "model",
        "training",
        "formal_shard_integrity",
        "calibration",
        "primary_monitor_evaluation",
        "artifacts",
        "gates",
    )
    return {field: config.get(field) for field in fields}


def validate_mixed_source_shard_set(
    target_protocol_path: str | Path,
    policy_manifest_path: str | Path,
    *,
    source_protocol_paths: Mapping[str, str | Path],
    shard_paths: Mapping[str, Sequence[str | Path]],
    partitions: Sequence[str] = FORMAL_PARTITIONS,
) -> dict[str, Any]:
    """Validate a relock whose partitions intentionally use old source files.

    Each partition is checked against the protocol that produced its shards;
    the target relock only supplies the scientific split declaration.  Source
    protocol hashes, immutable contracts, exact seed coverage, and cross-
    partition seed disjointness are all checked before a caller may train or
    calibrate a monitor.
    """
    target_file = Path(target_protocol_path).resolve()
    policy_file = Path(policy_manifest_path).resolve()
    requested = tuple(str(partition) for partition in partitions)
    errors: list[str] = []
    report: dict[str, Any] = {
        "gate_version": f"{GATE_VERSION}+mixed-source-v1",
        "status": "blocked",
        "passed": False,
        "target_protocol": {"path": str(target_file)},
        "policy_manifest": {"path": str(policy_file)},
        "partitions": {},
        "sources": {},
        "cross_partition": {"seed_overlap": {}},
        "errors": errors,
    }
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(partition not in FORMAL_PARTITIONS for partition in requested)
    ):
        errors.append("mixed-source gate received an invalid partition selection")
        return report
    required_partitions = set(requested)
    if set(source_protocol_paths) != required_partitions:
        errors.append("source_protocol_paths keys must match the requested partitions")
    if set(shard_paths) != required_partitions:
        errors.append("shard_paths keys must match the requested partitions")
    if errors:
        return report
    try:
        target = _load_json(target_file)
        _load_json(policy_file)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"cannot load mixed-source gate configuration: {error}")
        return report
    target_errors = validate_monitor_protocol(target)
    if target_errors:
        errors.extend(f"invalid target monitor protocol: {error}" for error in target_errors)
        return report
    target_hash = sha256_file(target_file)
    report["target_protocol"]["sha256"] = target_hash
    if target.get("relock_version") != "1.3":
        errors.append("mixed-source gate requires a v1.3 target relock protocol")
    else:
        parent_reference = target.get("parent_monitor_protocol")
        if not isinstance(parent_reference, str) or not parent_reference:
            errors.append("v1.3 target relock parent protocol is missing")
        else:
            parent_file = _resolve_protocol_reference(
                parent_reference, relative_to=target_file.parents[1]
            )
            if not parent_file.is_file():
                errors.append(f"v1.3 target relock parent protocol is missing: {parent_reference}")
            else:
                try:
                    parent_config = _load_json(parent_file)
                    parent_hash = sha256_file(parent_file)
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                    errors.append(f"cannot load v1.3 target relock parent: {error}")
                else:
                    relock_errors = validate_monitor_relock_protocol(
                        target, parent_config=parent_config, parent_sha256=parent_hash
                    )
                    if relock_errors:
                        errors.extend(
                            f"invalid target monitor relock: {error}"
                            for error in relock_errors
                        )
                    report["target_protocol"]["parent_path"] = str(parent_file)
                    report["target_protocol"]["parent_sha256"] = parent_hash

    source_configs: dict[str, dict[str, Any]] = {}
    source_contracts: dict[str, dict[str, Any]] = {}
    source_seed_sets: dict[str, set[int]] = {}
    for partition in requested:
        declared = str(source_protocol_paths[partition])
        if partition == "validation" and declared == "self":
            source_file = target_file
        else:
            source_file = _resolve_protocol_reference(
                declared, relative_to=target_file.parents[1]
            )
        source_record: dict[str, Any] = {
            "declared_path": declared,
            "path": str(source_file),
        }
        if not source_file.is_file():
            errors.append(f"source protocol for {partition} is missing: {declared}")
            report["sources"][partition] = source_record
            continue
        try:
            source_config = _load_json(source_file)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"cannot load source protocol for {partition}: {error}")
            report["sources"][partition] = source_record
            continue
        source_hash = sha256_file(source_file)
        source_record["sha256"] = source_hash
        report["sources"][partition] = source_record
        source_configs[partition] = source_config
        source_contracts[partition] = _protocol_contract(source_config)
        if partition == "validation" and source_file != target_file:
            errors.append("validation source must resolve to the target protocol via self")
        declaration = target.get("source_protocols", {}).get(partition, {})
        if not isinstance(declaration, Mapping):
            errors.append(f"target source_protocols.{partition} is missing")
        else:
            declared_target_path = str(declaration.get("path", ""))
            if declared_target_path != declared:
                errors.append(
                    f"target source_protocols.{partition}.path mismatch: "
                    f"expected={declared!r}, got={declared_target_path!r}"
                )
            if partition != "validation" and declaration.get("sha256") != source_hash:
                errors.append(
                    f"target source_protocols.{partition}.sha256 does not match source file"
                )
            if partition == "validation" and declared != "self":
                errors.append("target validation source declaration must be self")
    if source_contracts:
        baseline_partition = (
            "validation"
            if "validation" in source_contracts
            else next(iter(source_contracts))
        )
        baseline_contract = source_contracts[baseline_partition]
        for partition, contract in source_contracts.items():
            if contract != baseline_contract:
                differing = [
                    field
                    for field in baseline_contract
                    if contract.get(field) != baseline_contract.get(field)
                ]
                errors.append(
                    f"source protocol contract differs for {partition}: {differing}"
                )

    for partition in requested:
        source_file = report["sources"].get(partition, {}).get("path")
        if not source_file or partition not in source_configs:
            continue
        gate = validate_formal_shard_set(
            source_file,
            policy_file,
            partition=partition,
            shard_paths=shard_paths[partition],
        )
        report["partitions"][partition] = gate
        if not gate.get("passed"):
            errors.append(format_gate_failure(gate))
        seeds = {
            int(seed)
            for shard in gate.get("shards", [])
            for seed in shard.get("seeds", [])
        }
        source_seed_sets[partition] = seeds

    for index, first in enumerate(requested):
        for second in requested[index + 1 :]:
            overlap = sorted(
                source_seed_sets.get(first, set())
                & source_seed_sets.get(second, set())
            )
            report["cross_partition"]["seed_overlap"][f"{first}:{second}"] = overlap
            if overlap:
                errors.append(f"scene seeds overlap between {first} and {second}: {overlap}")
    report["cross_partition"]["seed_sets"] = {
        partition: sorted(seeds) for partition, seeds in source_seed_sets.items()
    }
    report["status"] = "passed" if not errors else "blocked"
    report["passed"] = not errors
    return report


def format_gate_failure(report: Mapping[str, Any], *, maximum_errors: int = 12) -> str:
    errors = [str(error) for error in report.get("errors", [])]
    shown = errors[:maximum_errors]
    suffix = f"\n- ... {len(errors) - len(shown)} more errors" if len(errors) > len(shown) else ""
    details = "\n".join(f"- {error}" for error in shown) or "- unspecified gate failure"
    return f"formal shard integrity gate blocked ({report.get('partition')}):\n{details}{suffix}"
