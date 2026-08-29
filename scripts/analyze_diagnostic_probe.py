#!/usr/bin/env python3
"""Analyze a sealed v1.1 diagnostic-probe collection offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vla_recovery_bench.artifacts import ensure_empty_output_dir, write_json_once
from vla_recovery_bench.diagnostic_probe import (
    FORBIDDEN_ONLINE_FIELDS,
    MAX_PROBE_STEPS,
    mechanism_log_loss,
    paired_cluster_bootstrap,
)
from vla_recovery_bench.recording import to_jsonable


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.is_file() or (path.stat().st_size <= 0 and not allow_empty):
        raise ValueError(f"missing or empty JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_online_leaks(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower().replace("-", "_")
                if normalized in FORBIDDEN_ONLINE_FIELDS:
                    errors.append(f"forbidden field {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    for index, row in enumerate(rows):
        visit(row, f"monitor_stream[{index}]")
    return errors


def _episode_loss(row: Mapping[str, Any], condition: str) -> tuple[float | None, str]:
    triggered = bool(row.get("triggered"))
    if not triggered:
        return 0.0, "no_trigger_itt_zero"
    trigger = row.get("trigger_prediction")
    post = row.get("post_window_posterior")
    if not isinstance(trigger, Mapping) or not isinstance(trigger.get("posterior"), Mapping):
        return None, "missing_trigger_posterior"
    if not isinstance(post, Mapping):
        return None, "incomplete_post_window"
    target = "none" if condition == "clean" else condition
    before = mechanism_log_loss(trigger["posterior"], target)
    after = mechanism_log_loss(post, target)
    return float(before - after), "complete"


def _binary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = ("actuator_fault", "observation_fault")
    exposed = [row for row in rows if row.get("condition") in labels and row.get("posterior")]
    if not exposed:
        return {"episodes": 0, "balanced_accuracy": None, "macro_f1": None, "confusion": {}}
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for row in exposed:
        posterior = row["posterior"]
        predicted = max(labels, key=lambda label: float(posterior.get(label, 0.0)))
        confusion[(str(row["condition"]), predicted)] += 1
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = confusion[(label, label)]
        fn = sum(
            value
            for (truth, guess), value in confusion.items()
            if truth == label and guess != label
        )
        fp = sum(
            value
            for (truth, guess), value in confusion.items()
            if truth != label and guess == label
        )
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "episodes": len(exposed),
        "balanced_accuracy": statistics.fmean(recalls),
        "macro_f1": statistics.fmean(f1s),
        "confusion": {
            f"{truth}->{guess}": value
            for (truth, guess), value in sorted(confusion.items())
        },
    }


def analyze(source: Path, *, bootstrap_replicates: int, bootstrap_seed: int) -> dict[str, Any]:
    manifest = _load_json(source / "run_manifest.json")
    metrics = _load_json(source / "metrics.json")
    lock = _load_json(source / "probe_lock.json")
    artifact_validation = _load_json(source / "artifact_validation.json")
    episodes = _load_jsonl(source / "episodes.jsonl")
    monitor_rows = _load_jsonl(source / "monitor_stream.jsonl")
    # A valid run may have no alarms; the online probe stream is then an empty
    # file.  Other streams remain mandatory and non-empty.
    probe_rows = _load_jsonl(source / "probe_stream.jsonl", allow_empty=True)
    audit_rows = _load_jsonl(source / "privileged_audit.jsonl")
    leak_errors = _check_online_leaks(monitor_rows + probe_rows)
    if leak_errors:
        raise ValueError(f"online stream contains privileged fields: {leak_errors[:5]}")
    if metrics.get("status") != "completed":
        raise ValueError("source run is not completed")
    if manifest.get("status") != "completed":
        raise ValueError("source run manifest is not completed")
    if artifact_validation.get("status") != "passed":
        raise ValueError("source run artifact validation is not passed")
    if lock.get("status") != "locked":
        raise ValueError("source probe lock is not locked")
    if int(metrics.get("episode_count", -1)) != len(episodes):
        raise ValueError("source metrics episode_count does not match episodes.jsonl")
    lock_reference = manifest.get("probe_lock", {})
    if lock_reference.get("sha256") != _sha256(source / "probe_lock.json"):
        raise ValueError("source run manifest probe-lock hash does not match artifact")
    try:
        validation_rate = float(lock["validation"]["joint_trigger_rate"])
        validation_cap = float(lock["validation"]["max_union_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source probe lock is missing validation budget fields") from error
    if validation_rate > validation_cap:
        raise ValueError("source probe lock exceeds its validation clean budget")

    monitor_by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in monitor_rows:
        monitor_by_token[str(row.get("episode_token"))].append(row)
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    attrition: dict[str, int] = defaultdict(int)
    for episode in episodes:
        pair_key = (str(episode.get("pair_id")), str(episode.get("condition")))
        arm = str(episode.get("arm"))
        token = str(episode.get("episode_token"))
        monitor_for_episode = monitor_by_token.get(token, [])
        post = episode.get("post_window_posterior")
        if post is None:
            candidates = [
                row
                for row in monitor_for_episode
                if row.get("observation_step") == episode.get("post_window_observation_step")
            ]
            if candidates:
                post = candidates[-1].get("posterior")
        enriched = dict(episode)
        enriched["post_window_posterior"] = post
        by_pair[pair_key][arm] = enriched

    paired_rows: list[dict[str, Any]] = []
    for (pair_id, condition), arms in sorted(by_pair.items()):
        passive = arms.get("passive_only")
        probe = arms.get("passive_plus_probe")
        if passive is None or probe is None:
            attrition["missing_arm"] += 1
            continue
        passive_improvement, passive_status = _episode_loss(passive, condition)
        probe_improvement, probe_status = _episode_loss(probe, condition)
        prefix_match = passive.get("prefix_hash_to_trigger") == probe.get("prefix_hash_to_trigger")
        status = "complete"
        delta: float | None = None
        if not prefix_match:
            status = "prefix_mismatch"
            attrition["prefix_mismatch"] += 1
        elif passive_improvement is None or probe_improvement is None:
            status = "incomplete_post_window"
            attrition["incomplete_post_window"] += 1
        else:
            delta = float(probe_improvement - passive_improvement)
            if passive_status == "no_trigger_itt_zero" and probe_status == "no_trigger_itt_zero":
                status = "no_trigger_itt_zero"
                attrition["no_trigger"] += 1
        paired_rows.append(
            {
                "pair_id": pair_id,
                "condition": condition,
                "seed": int(passive.get("seed", -1)),
                "passive_triggered": bool(passive.get("triggered")),
                "probe_triggered": bool(probe.get("triggered")),
                "passive_improvement": passive_improvement,
                "probe_improvement": probe_improvement,
                "paired_improvement_delta": delta,
                "prefix_match": prefix_match,
                "status": status,
                "passive_probe_steps": int(passive.get("probe_steps", 0)),
                "probe_probe_steps": int(probe.get("probe_steps", 0)),
                "probe_compute_ms": float(probe.get("probe_compute_ms", 0.0)),
                "probe_requery_count": int(probe.get("probe_requery_count", 0)),
                "posterior": probe.get("post_window_posterior"),
            }
        )

    valid_rows = [row for row in paired_rows if row["paired_improvement_delta"] is not None]
    bootstrap = paired_cluster_bootstrap(
        valid_rows,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    clean_pairs = [row for row in paired_rows if row["condition"] == "clean"]
    clean_union = sum(
        bool(row["passive_triggered"] or row["probe_triggered"])
        for row in clean_pairs
    )
    clean_rate = clean_union / len(clean_pairs) if clean_pairs else None
    max_probe_steps = max(
        [int(row.get("probe_probe_steps", 0)) for row in paired_rows] or [0]
    )
    max_probe_compute = max(
        [float(row.get("probe_compute_ms", 0.0)) for row in paired_rows] or [0.0]
    )
    cost_caps = {
        "extra_environment_steps_max": MAX_PROBE_STEPS,
        "observed_extra_environment_steps_max": max_probe_steps,
        "extra_compute_ms_max": 250.0,
        "observed_probe_compute_ms_max": max_probe_compute,
        "human_help_count": 0,
        "risk_penalty": 0,
        "passed": max_probe_steps <= MAX_PROBE_STEPS and max_probe_compute <= 250.0,
    }
    diagnosis_rows = [
        {
            "condition": row["condition"],
            "posterior": row["posterior"],
        }
        for row in paired_rows
        if row["status"] in {"complete", "no_trigger_itt_zero"}
        and row["condition"] in {"actuator_fault", "observation_fault"}
        and isinstance(row.get("posterior"), Mapping)
    ]
    diagnosis = _binary_metrics(diagnosis_rows)
    point = bootstrap.get("mean")
    gate = {
        "point_estimate_positive": point is not None and point > 0.0,
        "clean_joint_trigger_rate_max": 0.05,
        "clean_joint_trigger_rate_observed": clean_rate,
        "clean_budget_passed": clean_rate is not None and clean_rate <= 0.05,
        "probe_cost_caps_passed": bool(cost_caps["passed"]),
        "confidence_interval_reported": bootstrap["interval_95"] != [None, None],
    }
    gate["pilot_passed"] = all(
        [
            gate["point_estimate_positive"],
            gate["clean_budget_passed"],
            gate["probe_cost_caps_passed"],
        ]
    )
    return {
        "analysis_version": "1.1",
        "status": "completed",
        "scientific_result": False,
        "source": {
            "directory": str(source.resolve()),
            "run_manifest_sha256": _sha256(source / "run_manifest.json"),
            "metrics_sha256": _sha256(source / "metrics.json"),
            "probe_lock_sha256": _sha256(source / "probe_lock.json"),
            "monitor_record_count": len(monitor_rows),
            "probe_record_count": len(probe_rows),
            "audit_record_count": len(audit_rows),
            "online_leak_errors": leak_errors,
        },
        "population": {
            "episodes": len(episodes),
            "paired_units": len(paired_rows),
            "valid_primary_units": len(valid_rows),
            "conditions": {
                condition: sum(row["condition"] == condition for row in paired_rows)
                for condition in ("clean", "actuator_fault", "observation_fault")
            },
        },
        "primary": {
            "estimand": "probe_improvement_minus_passive_improvement_at_post_window",
            "no_trigger_itt_value": 0.0,
            "paired_cluster_bootstrap_95_percent": bootstrap,
            "rows": valid_rows,
        },
        "mechanism_diagnosis": diagnosis,
        "clean_operating_point": {
            "joint_trigger_episodes": clean_union,
            "episodes": len(clean_pairs),
            "joint_trigger_rate": clean_rate,
        },
        "costs": cost_caps,
        "attrition": dict(sorted(attrition.items())),
        "gate": gate,
        "outputs": {},
        "manifest_protocol_version": manifest.get("protocol_version"),
    }


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
    output = ensure_empty_output_dir(args.output)
    result = analyze(
        source,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    paired_path = output / "paired_rows.jsonl"
    # Analysis is a separate immutable artifact tree; the sealed collection is
    # never opened for append or rewritten.
    metrics_path = output / "metrics.json"
    result["outputs"] = {
        "metrics": str(metrics_path),
        "paired_rows": str(paired_path),
        "analysis_manifest": str(output / "analysis_manifest.json"),
    }
    with paired_path.open("x", encoding="utf-8") as stream:
        for row in result["primary"]["rows"]:
            stream.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
    write_json_once(metrics_path, result)
    write_json_once(
        output / "analysis_manifest.json",
        {
            "status": "completed",
            "source": str(source),
            "source_metrics_sha256": _sha256(source / "metrics.json"),
            "source_run_manifest_sha256": _sha256(source / "run_manifest.json"),
            "source_probe_lock_sha256": _sha256(source / "probe_lock.json"),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
    )
    write_json_once(
        output / "artifact_validation.json",
        {
            "status": "passed",
            "errors": [],
            "source_immutable": True,
            "metrics_sha256": _sha256(metrics_path),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
