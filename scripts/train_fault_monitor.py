#!/usr/bin/env python3
"""Train, calibrate, and validate the frozen-policy Phase-1 monitor only.

The GR00T checkpoint is never loaded by this command.  It reads the separated
monitor input channel, joins offline labels only inside the trainer, calibrates
on clean calibration episodes, and reports held-out validation metrics.  Final
test data is not accepted by this entry point.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vla_recovery_bench.artifacts import ensure_empty_output_dir, write_json_once
from vla_recovery_bench.monitor import (
    FEATURE_VERSION,
    MECHANISMS,
    FaultConditionedTemporalMonitor,
    mechanism_index,
    monitor_sha256,
)
from vla_recovery_bench.monitor_dataset import MonitorDatasetEpisode, load_monitor_dataset
from vla_recovery_bench.monitor_gate import (
    format_gate_failure,
    validate_formal_shard_set,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/monitor_training_v1_0.json"
DEFAULT_MANIFEST = ROOT / "configs/policies/groot_n1_5_robocasa_atomic_seen_30p.json"


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_provenance(directory: Path) -> dict[str, Any]:
    validation = _load_json(directory / "artifact_validation.json")
    if validation.get("status") != "passed":
        raise ValueError(f"source dataset did not pass artifact validation: {directory}")
    metrics = _load_json(directory / "metrics.json")
    before = _load_json(directory / "policy_state_before.json")
    after = _load_json(directory / "policy_state_after.json")
    if before.get("current_parameter_sha256") != after.get("current_parameter_sha256"):
        raise ValueError(f"source dataset changed the frozen policy: {directory}")
    return {
        "path": str(directory.resolve()),
        "partition": metrics.get("partition"),
        "debug": metrics.get("debug"),
        "collection_role": metrics.get("collection_role"),
        "episode_count": metrics.get("episode_count"),
        "rows": metrics.get("rows"),
        "monitor_inputs_sha256": _sha256(directory / "monitor_inputs.h5"),
        "offline_labels_sha256": _sha256(directory / "offline_labels.h5"),
        "shard_integrity_sha256": _sha256(directory / "shard_integrity.json"),
        "policy_parameter_sha256": before.get("current_parameter_sha256"),
    }


def _training_arrays(
    episodes: list[MonitorDatasetEpisode],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    episode_ids: list[str] = []
    for episode in episodes:
        features.append(episode.features)
        row_labels = np.full(
            episode.features.shape[0], mechanism_index("none"), dtype=np.int64
        )
        if episode.mechanism != "none":
            row_labels[episode.exposure] = mechanism_index(episode.mechanism)
        labels.append(row_labels)
        episode_ids.extend([episode.token] * episode.features.shape[0])
    return np.concatenate(features), np.concatenate(labels), episode_ids


def _load_sources(
    directories: list[Path], *, expected_partition: str
) -> tuple[list[MonitorDatasetEpisode], list[dict[str, Any]]]:
    if not directories:
        raise ValueError(f"at least one {expected_partition} dataset is required")
    resolved = [directory.resolve() for directory in directories]
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"duplicate {expected_partition} dataset directory")
    episodes: list[MonitorDatasetEpisode] = []
    provenance: list[dict[str, Any]] = []
    for directory in resolved:
        episodes.extend(
            load_monitor_dataset(directory, expected_partition=expected_partition)
        )
        provenance.append(_dataset_provenance(directory))
    tokens = [episode.token for episode in episodes]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"duplicate episode token across {expected_partition} shards")
    seed_conditions = [(episode.seed, episode.mechanism) for episode in episodes]
    if len(seed_conditions) != len(set(seed_conditions)):
        raise ValueError(
            f"duplicate scene-seed/mechanism pair across {expected_partition} shards"
        )
    return episodes, provenance


def _binary_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = ("actuator_fault", "observation_fault")
    confusion = Counter(zip(actual, predicted, strict=True))
    recalls: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        true_positive = confusion[(label, label)]
        false_negative = sum(
            count
            for (truth, guess), count in confusion.items()
            if truth == label and guess != label
        )
        false_positive = sum(
            count
            for (truth, guess), count in confusion.items()
            if truth != label and guess == label
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recalls.append(recall)
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "episodes": len(actual),
        "confusion": {
            f"{truth}->{guess}": count for (truth, guess), count in sorted(confusion.items())
        },
        "balanced_accuracy": statistics.fmean(recalls) if recalls else None,
        "macro_f1": statistics.fmean(f1_values) if f1_values else None,
    }


def evaluate_monitor(
    monitor: FaultConditionedTemporalMonitor,
    episodes: list[MonitorDatasetEpisode],
) -> dict[str, Any]:
    clean_episode_maxima: list[float] = []
    clean_alarm_episodes = 0
    clean_alarm_steps = 0
    clean_steps = 0
    actual: list[str] = []
    predicted: list[str] = []
    exposed_fault_episodes = 0
    detected_fault_episodes = 0
    delays: list[int] = []
    episode_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    prediction_steps = 0
    for episode in episodes:
        monitor.reset()
        rows = [monitor.predict_features(feature) for feature in episode.features]
        risks = np.asarray([float(row["risk"]) for row in rows], dtype=np.float64)
        alarms = risks >= monitor.threshold_
        prediction_steps += len(rows)
        record: dict[str, Any] = {
            "episode_token": episode.token,
            "pair_id": episode.pair_id,
            "seed": episode.seed,
            "mechanism": episode.mechanism,
            "steps": len(rows),
            "maximum_risk": float(risks.max()) if len(risks) else None,
            "alarm_steps": int(alarms.sum()),
            "success": episode.success,
        }
        if episode.mechanism == "none":
            maximum = float(risks.max()) if len(risks) else 0.0
            clean_episode_maxima.append(maximum)
            clean_alarm_episodes += int(bool(alarms.any()))
            clean_alarm_steps += int(alarms.sum())
            clean_steps += len(rows)
        elif episode.exposure.any():
            exposed_fault_episodes += 1
            window_indices = np.flatnonzero(episode.exposure)
            posterior = {
                name: float(
                    statistics.fmean(rows[index]["posterior"][name] for index in window_indices)
                )
                for name in MECHANISMS
            }
            diagnosis = max(
                ("actuator_fault", "observation_fault"), key=lambda name: posterior[name]
            )
            actual.append(episode.mechanism)
            predicted.append(diagnosis)
            window_alarm = alarms[window_indices]
            if window_alarm.any():
                detected_fault_episodes += 1
                delays.append(int(np.flatnonzero(window_alarm)[0]))
            record.update(
                {
                    "exposed": True,
                    "exposure_rows": int(len(window_indices)),
                    "mean_exposure_posterior": posterior,
                    "diagnosis": diagnosis,
                    "detected_in_exposure": bool(window_alarm.any()),
                    "detection_delay_steps": (
                        int(np.flatnonzero(window_alarm)[0]) if window_alarm.any() else None
                    ),
                }
            )
        else:
            record.update({"exposed": False, "not_exposed": True})
        episode_records.append(record)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    diagnosis = _binary_metrics(actual, predicted)
    return {
        "episodes": len(episodes),
        "independent_scene_seeds": len({episode.seed for episode in episodes}),
        "mechanism_diagnosis": diagnosis,
        "clean_operating_point": {
            "threshold": monitor.threshold_,
            "clean_episodes": len(clean_episode_maxima),
            "false_intervention_episodes": clean_alarm_episodes,
            "false_intervention_episode_rate": (
                clean_alarm_episodes / len(clean_episode_maxima) if clean_episode_maxima else None
            ),
            "alarm_steps_per_1000": (
                1000.0 * clean_alarm_steps / clean_steps if clean_steps else None
            ),
        },
        "exposure_window_detection": {
            "exposed_fault_episodes": exposed_fault_episodes,
            "detected_fault_episodes": detected_fault_episodes,
            "recall": (
                detected_fault_episodes / exposed_fault_episodes if exposed_fault_episodes else None
            ),
            "mean_delay_steps": statistics.fmean(delays) if delays else None,
        },
        "latency": {
            "total_ms": elapsed_ms,
            "prediction_steps": prediction_steps,
            "mean_ms_per_step": elapsed_ms / prediction_steps if prediction_steps else None,
        },
        "episode_records": episode_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-data", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration-data", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_directories = [*args.train_data, *args.calibration_data, *args.validation_data]
    if len(all_directories) != len({path.resolve() for path in all_directories}):
        raise ValueError(
            "train, calibration, and validation dataset directories must be distinct"
        )
    protocol = _load_json(args.protocol)
    formal_gate_reports = {
        "train": validate_formal_shard_set(
            args.protocol,
            args.manifest,
            partition="train",
            shard_paths=args.train_data,
        ),
        "calibration": validate_formal_shard_set(
            args.protocol,
            args.manifest,
            partition="calibration",
            shard_paths=args.calibration_data,
        ),
        "validation": validate_formal_shard_set(
            args.protocol,
            args.manifest,
            partition="validation",
            shard_paths=args.validation_data,
        ),
    }
    blocked = [
        format_gate_failure(report)
        for report in formal_gate_reports.values()
        if not report["passed"]
    ]
    if blocked:
        raise ValueError("\n\n".join(blocked))
    train, train_provenance = _load_sources(
        args.train_data, expected_partition="train"
    )
    calibration, calibration_provenance = _load_sources(
        args.calibration_data, expected_partition="calibration"
    )
    validation, validation_provenance = _load_sources(
        args.validation_data, expected_partition="validation"
    )
    train_seeds = {episode.seed for episode in train}
    calibration_seeds = {episode.seed for episode in calibration}
    validation_seeds = {episode.seed for episode in validation}
    if (
        train_seeds & calibration_seeds
        or train_seeds & validation_seeds
        or calibration_seeds & validation_seeds
    ):
        raise ValueError("monitor dataset scene-seed splits overlap")
    pilot_seeds = set(protocol["splits"]["pilot_seeds"])
    final_seeds = set(protocol["splits"]["final_test_scene_seeds"])
    if (train_seeds | calibration_seeds | validation_seeds) & (pilot_seeds | final_seeds):
        raise ValueError(
            "pilot or final-test seeds entered monitor training/calibration/validation"
        )

    output = ensure_empty_output_dir(args.output)
    checkpoint_path = output / "monitor.npz"
    sha_path = output / "monitor.sha256"
    calibration_path = output / "calibration.json"
    metrics_path = output / "metrics.json"
    manifest_path = output / "run_manifest.json"
    validation_path = output / "artifact_validation.json"
    episodes_path = output / "validation_episodes.jsonl"
    monitor = FaultConditionedTemporalMonitor(
        window_size=int(protocol["model"]["input_window_steps"]),
        seed=int(protocol["model"]["random_seed"]),
    )
    features, labels, episode_ids = _training_arrays(train)
    fit_report = monitor.fit(
        features,
        labels,
        episode_ids=episode_ids,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    clean_calibration = [episode.features for episode in calibration if episode.mechanism == "none"]
    calibration_report = monitor.calibrate_clean_episode_maxima(
        clean_calibration,
        false_intervention_rate=float(
            protocol["calibration"]["clean_false_intervention_budget"]["episode_rate_max"]
        ),
    )
    monitor.save(checkpoint_path)
    checkpoint_sha = monitor_sha256(checkpoint_path)
    with sha_path.open("x", encoding="utf-8") as stream:
        stream.write(f"{checkpoint_sha}  {checkpoint_path.name}\n")
    write_json_once(calibration_path, calibration_report)
    held_out = evaluate_monitor(monitor, validation)
    with episodes_path.open("x", encoding="utf-8") as stream:
        for record in held_out.pop("episode_records"):
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    source_provenance = {
        "train": train_provenance,
        "calibration": calibration_provenance,
        "validation": validation_provenance,
    }
    source_debug = any(
        bool(source["debug"])
        for sources in source_provenance.values()
        for source in sources
    )
    clean_rate = held_out["clean_operating_point"]["false_intervention_episode_rate"]
    budget = float(
        protocol["calibration"]["clean_false_intervention_budget"]["episode_rate_max"]
    )
    diagnosis_ba = held_out["mechanism_diagnosis"]["balanced_accuracy"]
    gate_passed = bool(
        not source_debug
        and clean_rate is not None
        and clean_rate <= budget
        and diagnosis_ba is not None
        and diagnosis_ba >= 0.65
    )
    metrics = {
        "status": "completed",
        "scientific_result": False,
        "debug": source_debug,
        "protocol_version": protocol["protocol_version"],
        "monitor_protocol_version": protocol["monitor_protocol_version"],
        "model": {
            "name": monitor.name,
            "checkpoint": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "feature_version": FEATURE_VERSION,
            "fit": fit_report,
            "policy_parameters_in_model": False,
        },
        "calibration": calibration_report,
        "held_out_validation": held_out,
        "sources": source_provenance,
        "formal_shard_integrity_gate": formal_gate_reports,
        "gate": {
            "status": "passed" if gate_passed else "blocked",
            "passed": gate_passed,
            "reason": (
                "held-out clean and mechanism criteria passed"
                if gate_passed
                else "debug data or held-out clean/mechanism criteria did not pass"
            ),
            "recovery_enabled": False,
            "diagnostic_probe_enabled": False,
            "final_test_accessed": False,
        },
    }
    write_json_once(metrics_path, metrics)
    write_json_once(
        manifest_path,
        {
            "protocol_version": protocol["protocol_version"],
            "monitor_protocol_version": protocol["monitor_protocol_version"],
            "status": "completed",
            "scientific_result": False,
            "debug": source_debug,
            "information_boundary": protocol["information_boundary"],
            "splits": {
                "train": sorted(train_seeds),
                "calibration": sorted(calibration_seeds),
                "validation": sorted(validation_seeds),
                "pilot_used": False,
                "final_test_used": False,
            },
            "monitor": metrics["model"],
            "calibration": calibration_report,
            "sources": source_provenance,
            "formal_shard_integrity_gate": formal_gate_reports,
            "protocol": {"path": str(args.protocol.resolve()), "sha256": _sha256(args.protocol)},
            "command": [sys.executable, *sys.argv],
        },
    )
    artifact_errors: list[str] = []
    for path in (
        checkpoint_path,
        sha_path,
        calibration_path,
        metrics_path,
        manifest_path,
        episodes_path,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            artifact_errors.append(f"missing or empty monitor artifact: {path}")
    loaded = FaultConditionedTemporalMonitor.load(checkpoint_path)
    if not loaded.fitted or monitor_sha256(checkpoint_path) != checkpoint_sha:
        artifact_errors.append("monitor checkpoint reload or SHA256 verification failed")
    write_json_once(
        validation_path,
        {"status": "passed" if not artifact_errors else "failed", "errors": artifact_errors},
    )
    if artifact_errors:
        raise RuntimeError(f"monitor artifact validation failed: {artifact_errors}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
