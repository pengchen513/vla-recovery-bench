#!/usr/bin/env python3
"""Audit a completed v1.4 pilot without modifying its immutable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vla_recovery_bench.artifacts import write_json_once
from vla_recovery_bench.pilot import validate_pilot_artifacts

MECHANISMS = ("actuator_fault", "observation_fault")
FORBIDDEN_MONITOR_FIELDS = {
    "condition",
    "episode_id",
    "executed_action",
    "fault",
    "fault_schedule",
    "info",
    "pair_id",
    "reward",
    "seed",
    "success",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_metrics(records: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    if not records:
        raise ValueError("binary metrics require at least one record")
    labels = MECHANISMS
    confusion = Counter((str(row["actual"]), str(row["predicted"])) for row in records)
    recalls: list[float] = []
    f1_values: list[float] = []
    per_class: dict[str, Any] = {}
    for label in labels:
        true_positive = confusion[(label, label)]
        false_negative = sum(confusion[(label, other)] for other in labels if other != label)
        false_positive = sum(confusion[(other, label)] for other in labels if other != label)
        support = true_positive + false_negative
        predicted = true_positive + false_positive
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        recalls.append(recall)
        f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "confusion": {
            f"{actual}->{predicted}": confusion[(actual, predicted)]
            for actual in labels
            for predicted in labels
        },
        "per_class": per_class,
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "macro_f1": float(sum(f1_values) / len(f1_values)),
    }


def _cluster_bootstrap(
    records: Sequence[Mapping[str, str]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["pair_id"])].append(record)
    clusters = sorted(grouped)
    if not clusters:
        raise ValueError("cluster bootstrap requires at least one pair_id")
    for cluster in clusters:
        actual = {str(record["actual"]) for record in grouped[cluster]}
        if actual != set(MECHANISMS):
            raise ValueError(f"cluster {cluster} does not contain both exposed mechanisms")

    generator = np.random.default_rng(seed)
    balanced_accuracy = np.empty(replicates, dtype=np.float64)
    macro_f1 = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.choice(clusters, size=len(clusters), replace=True)
        sample = [record for cluster in sampled for record in grouped[str(cluster)]]
        metrics = _binary_metrics(sample)
        balanced_accuracy[index] = metrics["balanced_accuracy"]
        macro_f1[index] = metrics["macro_f1"]
    return {
        "method": "scene_seed_cluster_percentile_bootstrap",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "independent_clusters": len(clusters),
        "balanced_accuracy": np.percentile(balanced_accuracy, [2.5, 97.5]).tolist(),
        "macro_f1": np.percentile(macro_f1, [2.5, 97.5]).tolist(),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _strict_exposure_records(
    episodes: Sequence[Mapping[str, Any]],
    monitor_by_token: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    observation_threshold: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for episode in episodes:
        condition = str(episode["condition"])
        if condition not in MECHANISMS or not int(episode.get("exposed_fault_count", 0)):
            continue
        token = str(episode["monitor_episode_token"])
        stream = monitor_by_token.get(token, ())
        step_key = (
            "returned_observation_step" if condition == "observation_fault" else "control_step"
        )
        onset = int(episode["first_affected_input_step"])
        end = int(episode["exposure_end_step_exclusive"])
        window = [record for record in stream if onset <= int(record[step_key]) < end]
        if len(window) != end - onset:
            errors.append(
                f"{episode['episode_id']} has {len(window)} monitor records in an "
                f"exposure window of length {end - onset}"
            )
            continue
        observation_evidence = [
            float(record["features"]["observation_evidence"]) for record in window
        ]
        alarms = [record for record in window if bool(record["decision"]["failure_detected"])]
        fault = episode["configured_faults"][0]
        parameters = fault.get("parameters", {})
        records.append(
            {
                "episode_id": str(episode["episode_id"]),
                "pair_id": str(episode["pair_id"]),
                "actual": condition,
                "predicted": (
                    "observation_fault"
                    if max(observation_evidence) >= observation_threshold
                    else "actuator_fault"
                ),
                "onset_step": onset,
                "duration_steps": end - onset,
                "variant": str(parameters.get("variant")),
                "camera_key": parameters.get("camera_key"),
                "maximum_observation_evidence": max(observation_evidence),
                "exposure_alarm": bool(alarms),
                "detection_delay_steps": (
                    int(alarms[0][step_key]) - onset if alarms else None
                ),
            }
        )
    return records, errors


def _condition_outcomes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        condition = str(episode["condition"])
        by_condition[condition].append(episode)
        by_pair[str(episode["pair_id"])][condition] = episode

    condition_summary = {
        condition: {
            "episodes": len(rows),
            "successes": sum(bool(row["success"]) for row in rows),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "mean_steps": float(np.mean([int(row["steps"]) for row in rows])),
        }
        for condition, rows in sorted(by_condition.items())
    }
    paired: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        differences = [
            int(bool(rows[mechanism]["success"])) - int(bool(rows["clean"]["success"]))
            for rows in by_pair.values()
        ]
        paired[mechanism] = {
            "pairs": len(differences),
            "mean_success_difference_vs_clean": float(np.mean(differences)),
            "fault_better": sum(value > 0 for value in differences),
            "clean_better": sum(value < 0 for value in differences),
            "tied": sum(value == 0 for value in differences),
        }
    all_conditions_identical = sum(
        len({bool(rows[condition]["success"]) for condition in ("clean", *MECHANISMS)}) == 1
        for rows in by_pair.values()
    )
    return {
        "conditions": condition_summary,
        "paired_vs_clean": paired,
        "all_three_conditions_same_success_outcome": all_conditions_identical,
        "scene_seed_pairs": len(by_pair),
    }


def analyze(
    source: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    episodes = _load_jsonl(source / "episodes.jsonl")
    monitor = _load_jsonl(source / "monitor_stream.jsonl")
    metrics = _load_json(source / "metrics.json")
    manifest = _load_json(source / "run_manifest.json")
    monitor_config = _load_json(source / "monitor_config.json")
    before = _load_json(source / "policy_state_before.json")
    after = _load_json(source / "policy_state_after.json")

    independent_artifact_errors = validate_pilot_artifacts(
        source, expected_episode_count=len(episodes)
    )
    monitor_by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    monitor_leaks: list[dict[str, Any]] = []
    for index, record in enumerate(monitor):
        leaked = sorted(FORBIDDEN_MONITOR_FIELDS & set(record))
        if leaked:
            monitor_leaks.append({"record_index": index, "fields": leaked})
        monitor_by_token[str(record.get("episode_token"))].append(record)

    observation_threshold = float(monitor_config["observation_diagnosis_threshold"])
    exposure, exposure_errors = _strict_exposure_records(
        episodes,
        monitor_by_token,
        observation_threshold=observation_threshold,
    )
    mechanism_metrics = _binary_metrics(exposure)
    bootstrap = _cluster_bootstrap(
        exposure,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    clean = [episode for episode in episodes if episode["condition"] == "clean"]
    clean_alarm_episodes = sum(int(episode["alarm_count"]) > 0 for episode in clean)
    clean_alarm_count = sum(int(episode["alarm_count"]) for episode in clean)
    clean_steps = sum(int(episode["steps"]) for episode in clean)
    detection_by_mechanism: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        rows = [record for record in exposure if record["actual"] == mechanism]
        delays = [
            int(record["detection_delay_steps"])
            for record in rows
            if record["detection_delay_steps"] is not None
        ]
        detection_by_mechanism[mechanism] = {
            "exposed_episodes": len(rows),
            "detected_within_exposure": len(delays),
            "exposure_window_recall": len(delays) / len(rows),
            "detection_delays_steps": delays,
            "mean_detection_delay_steps_among_detected": (
                float(np.mean(delays)) if delays else None
            ),
        }

    by_variant: dict[str, Any] = {}
    for actual, rows in _group_by(exposure, "actual").items():
        for variant, variant_rows in _group_by(rows, "variant").items():
            key = f"{actual}:{variant}"
            by_variant[key] = {
                "episodes": len(variant_rows),
                "correct": sum(row["actual"] == row["predicted"] for row in variant_rows),
                "exposure_window_alarms": sum(bool(row["exposure_alarm"]) for row in variant_rows),
            }

    reference = float(
        manifest.get("analysis", {}).get(
            "identifiability_reference_balanced_accuracy",
            0.5,
        )
    )
    minimum_gain = float(metrics["summary"]["exploratory_passive_rule"]["minimum_useful_gain"])
    balanced_ci = bootstrap["balanced_accuracy"]
    macro_f1_ci = bootstrap["macro_f1"]
    gate_checks = {
        "point_balanced_accuracy_gain_at_least_minimum": (
            mechanism_metrics["balanced_accuracy"] - reference >= minimum_gain
        ),
        "balanced_accuracy_ci_strictly_excludes_reference": balanced_ci[0] > reference,
        "macro_f1_ci_strictly_excludes_reference": macro_f1_ci[0] > reference,
        "held_out_fault_conditioned_monitor_evaluated": False,
        "calibrated_clean_operating_point_evaluated": False,
        "complete_future_action_chunk_available_in_source_stream": all(
            "requested_action_chunk" in record for record in monitor
        ),
        "diagnostic_probe_evaluated": False,
    }
    phase0_passed = all(gate_checks.values())

    source_files = (
        "run_manifest.json",
        "episodes.jsonl",
        "monitor_stream.jsonl",
        "audit_stream.jsonl",
        "metrics.json",
        "monitor_config.json",
        "policy_state_before.json",
        "policy_state_after.json",
        "artifact_validation.json",
    )
    outcome = _condition_outcomes(episodes)
    result = {
        "analysis_version": "1.1",
        "status": (
            "completed" if not independent_artifact_errors and not exposure_errors else "failed"
        ),
        "scientific_result": False,
        "source": {
            "directory": str(source),
            "protocol_version": manifest["protocol_version"],
            "source_metrics_status": metrics["status"],
            "artifact_validation_errors": independent_artifact_errors,
            "exposure_contract_errors": exposure_errors,
            "files_sha256": {name: _sha256(source / name) for name in source_files},
        },
        "population": {
            "episodes": len(episodes),
            "conditions": dict(Counter(str(row["condition"]) for row in episodes)),
            "exposed_fault_episodes": len(exposure),
            "exposed_scene_seed_clusters": len({row["pair_id"] for row in exposure}),
            "not_exposed_attrition": {
                mechanism: sum(
                    row["condition"] == mechanism and bool(row["not_exposed"])
                    for row in episodes
                )
                for mechanism in MECHANISMS
            },
        },
        "firewall": {
            "monitor_records": len(monitor),
            "forbidden_top_level_fields": sorted(FORBIDDEN_MONITOR_FIELDS),
            "leaked_record_count": len(monitor_leaks),
            "first_leaks": monitor_leaks[:10],
        },
        "policy_freeze": {
            "model_training_before": before["model_training"],
            "model_training_after": after["model_training"],
            "all_parameters_frozen_before": before["all_parameters_frozen"],
            "all_parameters_frozen_after": after["all_parameters_frozen"],
            "parameter_sha256_before": before["current_parameter_sha256"],
            "parameter_sha256_after": after["current_parameter_sha256"],
            "unchanged": (
                before["current_parameter_sha256"] == after["current_parameter_sha256"]
            ),
        },
        "fixed_transparent_rule": {
            "role": "pre_collection_evaluation_only_single_score_diagnostic",
            "scientific_result": False,
            "rule": (
                "predict observation_fault when maximum observation_evidence in the strict "
                f"exposure window is >= {observation_threshold}; otherwise actuator_fault"
            ),
            **mechanism_metrics,
            "reference_balanced_accuracy": reference,
            "minimum_useful_gain": minimum_gain,
            "cluster_bootstrap_95_percent": bootstrap,
            "by_variant": by_variant,
        },
        "clean_false_alarms": {
            "episodes": len(clean),
            "episodes_with_alarm": clean_alarm_episodes,
            "episode_rate": clean_alarm_episodes / len(clean),
            "episode_rate_wilson_95_percent": _wilson_interval(
                clean_alarm_episodes, len(clean)
            ),
            "alarm_events": clean_alarm_count,
            "clean_steps": clean_steps,
            "alarm_events_per_1000_steps": clean_alarm_count / clean_steps * 1000.0,
            "calibrated": False,
        },
        "strict_exposure_detection": detection_by_mechanism,
        "task_outcomes": outcome,
        "offline_oracle_upper_bound": {
            "role": "privileged_label_sanity_bound_only",
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "available_online": False,
        },
        "protocol_questions": {
            "passive_mechanism_identifiability": (
                "weak_signal_observed_but_not_established_by_a_held_out_monitor"
            ),
            "evidence_before_task_failure": (
                "not_resolved_no_preregistered_task_failure_onset_label"
            ),
            "diagnostic_probe_information_gain": "not_evaluated",
            "hard_nonshortcut_conditions": (
                "collected_but_no_recovery_intervention_effect_was_evaluated"
            ),
        },
        "phase0_gate": {
            "status": "passed" if phase0_passed else "not_passed",
            "checks": gate_checks,
            "interpretation": (
                "The fixed rule shows a weak passive mechanism signal, but its clean alarm "
                "behavior is unusable and the required held-out fault-conditioned monitor, "
                "calibrated operating point, and diagnostic-probe comparison are absent. "
                "Do not start confirmatory recovery experiments from this result."
            ),
            "next_required_work": (
                "freeze a diagnostic-probe protocol and a train/calibration/held-out split, then "
                "evaluate a fault-conditioned monitor at a controlled clean operating point"
            ),
        },
    }
    return result


def _group_by(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1404)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.parent == source:
        raise ValueError("analysis output must not modify the immutable source directory")
    result = analyze(
        source,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json_once(output, result)
    print(json.dumps({"output": str(output), **result}, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
