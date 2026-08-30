#!/usr/bin/env python3
"""Lock the v1.1 diagnostic-probe joint trigger under the v1.2 relock.

This command is offline and read-only with respect to all source artifacts.  It
does not start RoboCasa, contact a policy server, or access final-test seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from vla_recovery_bench.artifacts import ensure_empty_output_dir, write_json_once
from vla_recovery_bench.diagnostic_probe import (
    PROBE_PROTOCOL_VERSION,
    choose_entropy_threshold,
)
from vla_recovery_bench.monitor import FaultConditionedTemporalMonitor, monitor_sha256
from vla_recovery_bench.monitor_dataset import load_monitor_dataset
from vla_recovery_bench.monitor_gate import (
    validate_formal_shard_set,
    validate_mixed_source_shard_set,
)
from vla_recovery_bench.monitor_protocol import (
    validate_monitor_relock_protocol,
    validate_probe_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/monitor_relock_v1_2.json"
DEFAULT_PROBE = ROOT / "configs/diagnostic_probe_v1_1.json"
DEFAULT_MANIFEST = ROOT / "configs/policies/groot_n1_5_robocasa_atomic_seen_30p.json"
DEFAULT_MONITOR = Path("/home/pc/VLA/outputs/monitor_v1_0_formal_model/monitor.npz")
DEFAULT_MONITOR_METRICS = Path("/home/pc/VLA/outputs/monitor_v1_0_formal_model/metrics.json")


class LockBlockedError(ValueError):
    """A scientifically required lock gate failed before a lock was issued."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def _binomial_interval(successes: int, total: int, *, confidence: float = 0.95) -> dict[str, float]:
    """Return an exact two-sided Clopper–Pearson interval for a binomial rate."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("binomial successes/total are out of range")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    try:
        from scipy.stats import beta
    except ImportError as error:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "scipy is required to report Clopper-Pearson confidence intervals"
        ) from error
    alpha = 1.0 - confidence
    lower = (
        0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    )
    return {"lower": lower, "upper": upper, "confidence": confidence}


def _rate_report(successes: int, total: int, *, steps: int) -> dict[str, Any]:
    if total <= 0 or steps <= 0:
        raise ValueError("rate report requires positive episode and step counts")
    return {
        "episodes": int(successes),
        "total_episodes": int(total),
        "rate": float(successes / total),
        "clopper_pearson_95_percent": _binomial_interval(successes, total),
        "clean_steps": int(steps),
        "events_per_1000_clean_steps": float(1000.0 * successes / steps),
    }


def _validate_relock(protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Any] | None:
    """Validate relock metadata and return parent provenance when applicable."""
    if protocol.get("relock_version") is None:
        return None
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
        raise ValueError(f"monitor relock parent protocol is missing: {reference}")
    parent = _load_json(parent_path)
    parent_hash = _sha256(parent_path)
    errors = validate_monitor_relock_protocol(
        protocol, parent_config=parent, parent_sha256=parent_hash
    )
    if errors:
        raise ValueError(f"invalid monitor relock protocol: {errors}")
    return {"path": str(parent_path), "sha256": parent_hash}


def _resolve_protocol_reference(reference: str, *, protocol_path: Path) -> Path:
    candidate = Path(reference)
    candidates = (
        candidate,
        protocol_path.parents[1] / candidate,
        protocol_path.parent / candidate,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return (protocol_path.parents[1] / candidate).resolve()


def _resolve_lock_sources(
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    calibration_protocol: Path | None,
    validation_protocol: Path | None,
) -> tuple[dict[str, str], dict[str, Path]]:
    """Resolve source protocols and reject paths that drift from relock metadata."""
    if protocol.get("relock_version") not in {"1.3", "1.4"}:
        target = protocol_path.resolve()
        return (
            {"calibration": str(target), "validation": str(target)},
            {"calibration": target, "validation": target},
        )
    declarations = protocol.get("source_protocols", {})
    options = {"calibration": calibration_protocol, "validation": validation_protocol}
    declared: dict[str, str] = {}
    resolved: dict[str, Path] = {}
    for partition, option in options.items():
        path_decl = str(declarations.get(partition, {}).get("path", ""))
        if not path_decl:
            raise ValueError(
                f"v{protocol.get('relock_version')} source_protocols.{partition}.path is missing"
            )
        expected = (
            protocol_path.resolve()
            if path_decl == "self"
            else _resolve_protocol_reference(path_decl, protocol_path=protocol_path)
        )
        actual = expected if option is None else option.resolve()
        if actual != expected:
            raise ValueError(
                f"{partition} protocol does not match relock declaration: "
                f"expected={expected}, got={actual}"
            )
        declared[partition] = path_decl
        resolved[partition] = actual
    return declared, resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_summaries(
    monitor: FaultConditionedTemporalMonitor,
    directories: list[Path],
    *,
    clean_only: bool = True,
) -> list[dict[str, Any]]:
    episodes = []
    partition = "calibration" if clean_only else "validation"
    for directory in directories:
        episodes.extend(load_monitor_dataset(directory, expected_partition=partition))
    summaries: list[dict[str, Any]] = []
    for episode in sorted(episodes, key=lambda item: item.token):
        if episode.mechanism != "none":
            continue
        monitor.reset()
        predictions = [monitor.predict_features(row) for row in episode.features]
        if not predictions:
            raise ValueError(f"clean episode has no monitor rows: {episode.token}")
        summaries.append(
            {
                "episode_id": episode.episode_id or episode.token,
                "episode_token": episode.token,
                "seed": episode.seed,
                "maximum_risk": max(float(row["risk"]) for row in predictions),
                "maximum_entropy": max(float(row["normalized_entropy"]) for row in predictions),
                "steps": len(predictions),
            }
        )
    return summaries


def lock_threshold(
    *,
    protocol_path: Path,
    probe_path: Path,
    manifest_path: Path,
    monitor_path: Path,
    monitor_metrics_path: Path,
    calibration_dirs: list[Path],
    validation_dirs: list[Path],
    output: Path,
    calibration_protocol_path: Path | None = None,
    validation_protocol_path: Path | None = None,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    probe = _load_json(probe_path)
    _load_json(manifest_path)
    parent_provenance = _validate_relock(protocol_path, protocol)
    errors = validate_probe_protocol(probe, protocol)
    if errors:
        raise ValueError(f"invalid v1.1 probe protocol: {errors}")
    if probe.get("probe_protocol_version") != PROBE_PROTOCOL_VERSION:
        raise ValueError("probe protocol version mismatch")
    if not calibration_dirs or not validation_dirs:
        raise ValueError("calibration and validation shard directories are required")

    declared_sources, resolved_sources = _resolve_lock_sources(
        protocol_path=protocol_path,
        protocol=protocol,
        calibration_protocol=calibration_protocol_path,
        validation_protocol=validation_protocol_path,
    )
    if protocol.get("relock_version") == "1.3":
        mixed_gate = validate_mixed_source_shard_set(
            protocol_path,
            manifest_path,
            source_protocol_paths=declared_sources,
            shard_paths={
                "calibration": calibration_dirs,
                "validation": validation_dirs,
            },
            partitions=("calibration", "validation"),
        )
        calibration_gate = mixed_gate["partitions"].get("calibration", {})
        validation_gate = mixed_gate["partitions"].get("validation", {})
        if not mixed_gate["passed"]:
            raise ValueError(
                "mixed-source formal shard gate failed before entropy lock: "
                + "\n".join(f"- {error}" for error in mixed_gate["errors"][:20])
            )
    else:
        mixed_gate = None
        calibration_gate = validate_formal_shard_set(
            resolved_sources["calibration"],
            manifest_path,
            partition="calibration",
            shard_paths=calibration_dirs,
        )
        validation_gate = validate_formal_shard_set(
            resolved_sources["validation"],
            manifest_path,
            partition="validation",
            shard_paths=validation_dirs,
        )
    if not calibration_gate["passed"] or not validation_gate["passed"]:
        raise ValueError(
            "formal shard gate failed before entropy lock: "
            f"calibration={calibration_gate}, validation={validation_gate}"
        )

    metrics = _load_json(monitor_metrics_path)
    if metrics.get("status") != "completed" or not metrics.get("gate", {}).get("passed"):
        raise ValueError("monitor metrics do not report a passed completed gate")
    expected_monitor_hash = str(metrics.get("model", {}).get("sha256", ""))
    actual_monitor_hash = monitor_sha256(monitor_path)
    if expected_monitor_hash and expected_monitor_hash != actual_monitor_hash:
        raise ValueError(
            "monitor checkpoint hash does not match the passed monitor metrics: "
            f"expected={expected_monitor_hash}, got={actual_monitor_hash}"
        )
    monitor = FaultConditionedTemporalMonitor.load(monitor_path)
    risk_threshold = float(monitor.threshold_)
    expected_calibration_episodes = 3 * len(
        protocol.get("splits", {}).get("calibration_scene_seeds", ())
    )
    expected_validation_episodes = 3 * len(
        protocol.get("splits", {}).get("validation_scene_seeds", ())
    )
    # Each scene seed contributes exactly one clean row to the threshold
    # calculation.  The formal shard gate has already checked the three-way
    # condition cross, so derive this count from the frozen protocol rather
    # than silently assuming a particular future sample size.
    expected_calibration_clean = expected_calibration_episodes // 3
    expected_validation_clean = expected_validation_episodes // 3
    calibration = _prediction_summaries(monitor, calibration_dirs, clean_only=True)
    if len(calibration) != expected_calibration_clean:
        raise ValueError(
            "unexpected clean calibration episode count: "
            f"expected={expected_calibration_clean}, got={len(calibration)}"
        )
    lock = choose_entropy_threshold(
        calibration,
        risk_threshold=risk_threshold,
        max_union_rate=float(probe["entropy_lock"]["union_episode_rate_max"]),
    )

    validation = _prediction_summaries(monitor, validation_dirs, clean_only=False)
    if len(validation) != expected_validation_clean:
        raise ValueError(
            "unexpected clean validation episode count: "
            f"expected={expected_validation_clean}, got={len(validation)}"
        )

    calibration_seeds = sorted({int(row["seed"]) for row in calibration})
    validation_seeds = sorted({int(row["seed"]) for row in validation})
    pilot_seeds = set(range(500, 512))
    overlap = sorted((set(calibration_seeds) | set(validation_seeds)) & pilot_seeds)
    if protocol.get("relock_version") == "1.4" and overlap:
        raise ValueError(f"v1.4 fresh threshold data overlaps the frozen pilot seeds: {overlap}")
    if set(calibration_seeds) & set(validation_seeds):
        raise ValueError("calibration and validation clean seeds overlap")
    validation_risk = sum(row["maximum_risk"] >= risk_threshold for row in validation)
    validation_joint = sum(
        row["maximum_risk"] >= risk_threshold or row["maximum_entropy"] >= lock["entropy_threshold"]
        for row in validation
    )
    validation_rate = validation_joint / len(validation)
    max_rate = float(probe["entropy_lock"]["union_episode_rate_max"])
    if validation_rate > max_rate:
        raise LockBlockedError(
            "locked joint trigger exceeds the clean validation budget: "
            f"rate={validation_rate}, max={max_rate}",
            details={
                "validation_clean_episode_count": len(validation),
                "validation_risk_trigger_episodes": int(validation_risk),
                "validation_joint_trigger_episodes": int(validation_joint),
                "validation_joint_trigger_rate": float(validation_rate),
                "max_union_rate": max_rate,
                "risk_threshold": risk_threshold,
                "entropy_threshold": float(lock["entropy_threshold"]),
                "calibration_lock": lock,
                "validation_episode_summaries": validation,
                "validation_joint_trigger_rate_ci_95": _binomial_interval(
                    validation_joint, len(validation)
                ),
            },
        )

    calibration_joint = int(lock["joint_trigger_episodes"])
    calibration_steps = sum(int(row["steps"]) for row in calibration)
    validation_steps = sum(int(row["steps"]) for row in validation)

    result = {
        "status": "locked",
        "scientific_result": False,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "protocol": {"path": str(protocol_path.resolve()), "sha256": _sha256(protocol_path)},
        "probe_protocol": {"path": str(probe_path.resolve()), "sha256": _sha256(probe_path)},
        "policy_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
        },
        "monitor": {
            "path": str(monitor_path.resolve()),
            "sha256": actual_monitor_hash,
            "risk_threshold": risk_threshold,
            "metrics_path": str(monitor_metrics_path.resolve()),
        },
        "calibration_gate": calibration_gate,
        "validation_gate": validation_gate,
        "source_protocols": {
            partition: {
                "path": str(path),
                "sha256": _sha256(path),
                "declared_path": declared_sources[partition],
            }
            for partition, path in resolved_sources.items()
        },
        "mixed_source_gate": mixed_gate,
        "calibration": {**lock, "episode_summaries": calibration},
        "validation": {
            "clean_episode_count": len(validation),
            "risk_trigger_episodes": int(validation_risk),
            "joint_trigger_episodes": int(validation_joint),
            "joint_trigger_rate": float(validation_rate),
            "max_union_rate": max_rate,
            "episode_summaries": validation,
        },
        "relock": {
            "version": protocol.get("relock_version"),
            "parent_monitor_protocol": parent_provenance,
            "threshold_rule": protocol.get("relock_decisions", {}).get("threshold_rule"),
            "point_gate": protocol.get("relock_decisions", {}).get("validation_point_gate"),
            "confidence_interval": protocol.get("relock_decisions", {}).get(
                "validation_confidence_interval"
            ),
            "monitor_retraining": protocol.get("monitor_retraining"),
            "fresh_data_policy": protocol.get("fresh_data_policy"),
        },
        "data_independence": {
            "calibration_clean_seeds": calibration_seeds,
            "validation_clean_seeds": validation_seeds,
            "pilot_seed_range": [500, 511],
            "pilot_seed_overlap": overlap,
            "pilot_data_used_for_threshold": False,
            "threshold_selection_source": (
                "fresh_calibration_clean_episodes_only"
                if protocol.get("relock_version") == "1.4"
                else "declared_calibration_clean_episodes_only"
            ),
            "validation_source": "fresh_validation_clean_episodes_only",
        },
        "rate_reports": {
            "calibration_joint_trigger": _rate_report(
                calibration_joint, len(calibration), steps=calibration_steps
            ),
            "validation_joint_trigger": _rate_report(
                validation_joint, len(validation), steps=validation_steps
            ),
        },
        "formula": {
            "normalized_entropy": "-sum(p_i*log(p_i))/log(3)",
            "trigger": "maximum_risk >= risk_threshold OR maximum_entropy >= entropy_threshold",
            "budget_unit": "clean_episode",
            "threshold_locked_before_pilot": True,
            "confidence_interval_report_only": True,
        },
        "source_shards": {
            "calibration": [
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path / "monitor_inputs.h5"),
                    "artifacts": {
                        "monitor_inputs.h5": _sha256(path / "monitor_inputs.h5"),
                        "offline_labels.h5": _sha256(path / "offline_labels.h5"),
                    },
                }
                for path in calibration_dirs
            ],
            "validation": [
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path / "monitor_inputs.h5"),
                    "artifacts": {
                        "monitor_inputs.h5": _sha256(path / "monitor_inputs.h5"),
                        "offline_labels.h5": _sha256(path / "offline_labels.h5"),
                    },
                }
                for path in validation_dirs
            ],
        },
    }
    output_dir = ensure_empty_output_dir(output)
    write_json_once(output_dir / "probe_lock.json", result)
    write_json_once(
        output_dir / "artifact_validation.json",
        {"status": "passed", "errors": [], "lock_sha256": _sha256(output_dir / "probe_lock.json")},
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--monitor", type=Path, default=DEFAULT_MONITOR)
    parser.add_argument("--monitor-metrics", type=Path, default=DEFAULT_MONITOR_METRICS)
    parser.add_argument("--calibration-protocol", type=Path, default=None)
    parser.add_argument("--validation-protocol", type=Path, default=None)
    parser.add_argument("--calibration-data", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/pc/VLA/outputs/diagnostic_probe_v1_2_lock"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = lock_threshold(
            protocol_path=args.protocol,
            probe_path=args.probe,
            manifest_path=args.manifest,
            monitor_path=args.monitor,
            monitor_metrics_path=args.monitor_metrics,
            calibration_dirs=args.calibration_data,
            validation_dirs=args.validation_data,
            output=args.output,
            calibration_protocol_path=args.calibration_protocol,
            validation_protocol_path=args.validation_protocol,
        )
    except Exception as error:
        # Preserve a machine-readable explanation of a blocked lock.  The
        # output directory is write-once; an existing directory is never
        # replaced or amended, so a rerun must choose a new path explicitly.
        failure: dict[str, Any] = {
            "status": "blocked",
            "scientific_result": False,
            "probe_protocol_version": PROBE_PROTOCOL_VERSION,
            "error_type": type(error).__name__,
            "error": str(error),
            "protocol": str(args.protocol.resolve()),
            "probe_protocol": str(args.probe.resolve()),
            "policy_manifest": str(args.manifest.resolve()),
            "monitor": str(args.monitor.resolve()),
            "monitor_metrics": str(args.monitor_metrics.resolve()),
            "calibration_shards": [str(path.resolve()) for path in args.calibration_data],
            "validation_shards": [str(path.resolve()) for path in args.validation_data],
            "calibration_protocol": (
                str(args.calibration_protocol.resolve())
                if args.calibration_protocol is not None
                else None
            ),
            "validation_protocol": (
                str(args.validation_protocol.resolve())
                if args.validation_protocol is not None
                else None
            ),
            "next_action": (
                "recalibrate or revise the pre-registered entropy lock after scientific review; "
                "do not run the diagnostic probe until a status=locked artifact exists"
            ),
        }
        for label, path in (
            ("protocol_sha256", args.protocol),
            ("probe_protocol_sha256", args.probe),
            ("policy_manifest_sha256", args.manifest),
            ("monitor_sha256", args.monitor),
            ("monitor_metrics_sha256", args.monitor_metrics),
        ):
            if path.is_file():
                failure[label] = _sha256(path)
        details = getattr(error, "details", None)
        if isinstance(details, dict):
            failure["details"] = details
        try:
            output_dir = ensure_empty_output_dir(args.output)
            write_json_once(output_dir / "probe_lock_failed.json", failure)
            write_json_once(
                output_dir / "artifact_validation.json",
                {
                    "status": "failed",
                    "errors": [str(error)],
                    "lock_issued": False,
                    "failure_artifact": str(output_dir / "probe_lock_failed.json"),
                },
            )
        except FileExistsError as directory_error:
            failure["artifact_write_error"] = str(directory_error)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        if isinstance(error, LockBlockedError):
            return 2
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
