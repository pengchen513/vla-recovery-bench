#!/usr/bin/env python3
"""Validate the frozen, metadata-only v1.4 research plan.

This command intentionally does not import RoboCasa, load a checkpoint, or run
an experiment.  It catches protocol drift before an experiment command is
allowed to create scientific artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from vla_recovery_bench.monitor_protocol import (
    validate_monitor_protocol,
    validate_probe_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "configs/identifiability_pilot_v1_4.json"
POWER_PATH = ROOT / "configs/power_analysis_v1_4.json"
INTERVENTION_PATH = ROOT / "configs/intervention_protocol_v1_4.json"
MONITOR_PATH = ROOT / "configs/monitor_training_v1_0.json"
PROBE_PATH = ROOT / "configs/diagnostic_probe_v1_0.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_pilot(pilot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if pilot.get("protocol_version") != "1.4":
        errors.append("pilot protocol_version must be 1.4")
    seeds = pilot.get("seed_block", {}).get("seeds", [])
    assignments = pilot.get("scene_seed_factors", [])
    if len(seeds) != 12 or len(set(seeds)) != 12:
        errors.append("pilot must contain exactly 12 unique scene seeds")
    if len(assignments) != 12:
        errors.append("pilot must contain exactly 12 scene-seed factor rows")
    if pilot.get("seed_block", {}).get("episodes_per_seed") != 3:
        errors.append("pilot episodes_per_seed must be 3")
    if pilot.get("episode_count") != 36:
        errors.append("pilot episode_count must be 36")
    timing = pilot.get("timing_contract", {})
    if timing.get("actuator_fault_phase") != "before_action":
        errors.append("actuator fault phase must be before_action")
    if timing.get("observation_fault_phase") != "after_step":
        errors.append("observation fault phase must be after_step")
    if timing.get("observation_first_affected_input_offset_steps") != 1:
        errors.append("observation first affected input offset must be one step")
    if timing.get("observation_injection_step_rule") != "onset_step - 1":
        errors.append("observation injection step rule must be onset_step - 1")
    assignment_seeds = [row.get("seed") for row in assignments]
    if assignment_seeds != list(seeds):
        errors.append("pilot scene-seed rows must exactly follow seed_block order")
    allowed_actuator = {"arm_only", "gripper_only"}
    allowed_observation = {"partial_mask", "blur", "stale_frame", "color_shift"}
    cameras = {
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    }
    for row in assignments:
        if row.get("actuator_variant") not in allowed_actuator:
            errors.append(f"unsupported pilot actuator variant: {row.get('actuator_variant')}")
        if row.get("observation_variant") not in allowed_observation:
            errors.append(
                f"unsupported pilot observation variant: {row.get('observation_variant')}"
            )
        if row.get("observation_camera") not in cameras:
            errors.append(f"unsupported pilot camera: {row.get('observation_camera')}")
        if row.get("onset_step") not in {80, 240}:
            errors.append(f"pilot onset must be 80 or 240: {row.get('onset_step')}")
        if row.get("duration_steps") not in {4, 8}:
            errors.append(f"pilot duration must be 4 or 8: {row.get('duration_steps')}")
    balance = pilot.get("balance_requirements", {})
    expected_counts = {
        "onset_step": {80: 6, 240: 6},
        "duration_steps": {4: 6, 8: 6},
        "actuator_variant": {"arm_only": 6, "gripper_only": 6},
        "observation_variant": {
            "partial_mask": 3,
            "blur": 3,
            "stale_frame": 3,
            "color_shift": 3,
        },
    }
    for field, expected in expected_counts.items():
        observed = Counter(row.get(field) for row in assignments)
        if dict(observed) != expected:
            errors.append(f"pilot {field} must be balanced as {expected}, got {dict(observed)}")
    cameras = {
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    }
    camera_counts = Counter(row.get("observation_camera") for row in assignments)
    if set(camera_counts) != cameras or any(camera_counts[camera] != 4 for camera in cameras):
        errors.append(
            "pilot cameras must each occur exactly four times, "
            f"got {dict(camera_counts)}"
        )
    if balance.get("each_scene_seed_reused_across_fault_mechanisms") is not True:
        errors.append("each scene seed must be reused across both fault mechanisms")
    if pilot.get("analysis", {}).get("clean_use") != "false_intervention_and_calibration_only":
        errors.append(
            "clean episodes may only support false-intervention and "
            "calibration estimates"
        )
    if pilot.get("analysis", {}).get("all_zero_conditions_are_primary_evidence") is not False:
        errors.append("all-zero conditions must not be primary evidence")
    if pilot.get("analysis", {}).get("pilot_is_recovery_evidence") is not False:
        errors.append("pilot must not be treated as recovery evidence")
    if pilot.get("analysis", {}).get("not_exposed_before_onset_is_attrition") is not True:
        errors.append("early termination before exposure must be reported as attrition")
    return errors


def validate_power(power: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if power.get("protocol_version") != "1.4":
        errors.append("power protocol_version must be 1.4")
    expected = {
        "alpha_two_sided": 0.05,
        "target_power": 0.8,
        "minimum_detectable_absolute_effect": 0.2,
        "baseline_recovery_rate": 0.5,
        "within_pair_correlation": 0.3,
        "invalid_run_rate": 0.05,
        "attrition_allowance": 0.1,
    }
    for key, value in expected.items():
        if power.get(key) != value:
            errors.append(f"power {key} must remain frozen at {value}")
    sample_rule = power.get("sample_size_rule", {})
    if sample_rule.get("method") != "deterministic_monte_carlo_required":
        errors.append("power sample size must use deterministic Monte Carlo")
    if sample_rule.get("replicates") != 10000:
        errors.append("power simulation must use 10000 replicates")
    candidates = sample_rule.get("candidate_independent_units", [])
    if not candidates or any(int(n) <= 0 or int(n) % 12 for n in candidates):
        errors.append("power candidates must be positive multiples of 12")
    if len(sample_rule.get("sensitivity_scenarios", [])) < 4:
        errors.append("power must include at least four sensitivity scenarios")
    return errors


def validate_intervention(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("protocol_version") != "1.4":
        errors.append("intervention protocol_version must be 1.4")
    assignment = protocol.get("assignment", {})
    if assignment.get("randomization_seed") != (
        "sha256(pair_id || decision_step || protocol_version)"
    ):
        errors.append("common-prefix randomization rule drifted")
    if assignment.get("primary_arms") != {"selector": 0.5, "fixed_retry": 0.5}:
        errors.append("primary intervention arms must be balanced selector/fixed_retry")
    costs = protocol.get("cost_vector", {})
    required = {
        "elapsed_execution_time_ms",
        "extra_compute_ms",
        "peak_memory_mb",
        "human_help_count",
        "risk_penalty",
    }
    if set(costs) != required:
        errors.append("cost vector must contain exactly the five declared components")
    if protocol.get("analysis", {}).get("clean_budget_is_event_count") is not False:
        errors.append("clean budget cannot be represented only as event count")
    if protocol.get("implementation_gate", {}).get("state_clone_test") != (
        "required_for_common_prefix_causal_claim"
    ):
        errors.append("state clone test must gate causal common-prefix claims")
    primary = protocol.get("primary_comparison", {})
    if primary.get("primary_endpoint") != "delta_recovery_at_componentwise_cost_budget":
        errors.append("intervention protocol must declare exactly one recovery primary endpoint")
    if protocol.get("analysis", {}).get("scalar_cost") != "secondary_only":
        errors.append("scalar cost must remain secondary")
    return errors


def validate_all() -> list[str]:
    errors = []
    pilot = _load(PILOT_PATH)
    monitor = _load(MONITOR_PATH)
    probe = _load(PROBE_PATH)
    errors.extend(validate_pilot(pilot))
    errors.extend(validate_power(_load(POWER_PATH)))
    errors.extend(validate_intervention(_load(INTERVENTION_PATH)))
    errors.extend(validate_monitor_protocol(monitor))
    errors.extend(validate_probe_protocol(probe, monitor))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args()
    errors = validate_all()
    report = {
        "status": "passed" if not errors else "blocked",
        "protocol_version": "1.4",
        "scientific_result": False,
        "validated_files": [
            str(PILOT_PATH),
            str(POWER_PATH),
            str(INTERVENTION_PATH),
            str(MONITOR_PATH),
            str(PROBE_PATH),
        ],
        "errors": errors,
    }
    output = (
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else json.dumps(report, indent=2)
    )
    print(output)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
