"""Frozen Phase-1 monitor/probe protocol validation and episode planning."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .faults import FaultSchedule
from .types import FaultPhase, FaultSpec

PARTITIONS = ("train", "calibration", "validation", "final_test")
SPLIT_KEYS = {
    "pilot": "pilot_seeds",
    "train": "train_scene_seeds",
    "calibration": "calibration_scene_seeds",
    "validation": "validation_scene_seeds",
    "final_test": "final_test_scene_seeds",
}
CONDITIONS = ("clean", "actuator_fault", "observation_fault")
REQUIRED_ALLOWED_INPUTS = {
    "previous_observation",
    "current_observation",
    "requested_action_chunk",
    "bounded_requested_action_history",
    "task_instruction",
    "declared_wall_clock_latency",
}
REQUIRED_FORBIDDEN_INPUTS = {
    "fault_type",
    "fault_schedule",
    "fault_onset",
    "fault_duration",
    "executed_action",
    "reward",
    "success",
    "terminated",
    "truncated",
    "info",
    "mujoco_state",
    "final_test_labels",
}


def _sha_order(seed: int, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{seed}".encode()).hexdigest()


def _balanced_assign(
    seeds: Sequence[int], values: Sequence[Any], *, salt: str
) -> dict[int, Any]:
    if not values:
        raise ValueError(f"factor {salt} has no values")
    ordered = sorted((int(seed) for seed in seeds), key=lambda seed: _sha_order(seed, salt=salt))
    return {seed: values[index % len(values)] for index, seed in enumerate(ordered)}


def validate_monitor_protocol(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("protocol_version") != "1.4":
        errors.append("monitor protocol must target research protocol 1.4")
    if config.get("monitor_protocol_version") != "1.0":
        errors.append("monitor_protocol_version must be 1.0")
    boundary = config.get("information_boundary", {})
    if set(boundary.get("allowed_inputs", [])) != REQUIRED_ALLOWED_INPUTS:
        errors.append("monitor allowed_inputs drifted from the frozen information boundary")
    if set(boundary.get("forbidden_inputs", [])) != REQUIRED_FORBIDDEN_INPUTS:
        errors.append("monitor forbidden_inputs drifted from the frozen information boundary")
    if boundary.get("monitor_does_not_modify_policy") is not True:
        errors.append("monitor must not modify policy parameters")

    splits = config.get("splits", {})
    seen: dict[int, str] = {}
    for name, key in SPLIT_KEYS.items():
        raw = splits.get(key, [])
        if not isinstance(raw, list) or not raw:
            errors.append(f"split {key} must be a non-empty list")
            continue
        values = [int(seed) for seed in raw]
        if len(values) != len(set(values)):
            errors.append(f"split {key} contains duplicate seeds")
        if values != sorted(values):
            errors.append(f"split {key} must be sorted for reproducibility")
        for seed in values:
            prior = seen.get(seed)
            if prior is not None:
                errors.append(f"seed {seed} appears in both {prior} and {name}")
            seen[seed] = name
    expected_lengths = {
        "pilot": 12,
        "train": 100,
        "calibration": 50,
        "validation": 50,
        "final_test": 30,
    }
    for name, expected in expected_lengths.items():
        key = SPLIT_KEYS[name]
        if len(splits.get(key, [])) != expected:
            errors.append(f"split {key} must contain exactly {expected} scene seeds")

    fault = config.get("fault_sampling", {})
    if tuple(fault.get("mechanisms", ())) != CONDITIONS:
        errors.append("fault mechanisms must remain clean/actuator/observation in order")
    for field in (
        "onset_steps",
        "duration_steps",
        "actuator_variants",
        "observation_variants",
        "camera_variants",
    ):
        if not fault.get(field):
            errors.append(f"fault_sampling.{field} must not be empty")
    if "all_channel_zero" in set(fault.get("actuator_variants", [])):
        errors.append("all-channel-zero cannot enter primary monitor training")
    if "all_zero" in set(fault.get("observation_variants", [])):
        errors.append("all-zero images cannot enter primary monitor training")

    design = config.get("collection_design", {})
    if tuple(design.get("conditions_per_scene_seed", ())) != CONDITIONS:
        errors.append("each scene seed must cross all three conditions")
    if design.get("paired_factor_row_across_fault_mechanisms") is not True:
        errors.append("actuator and observation episodes must share the same factor row")
    storage = config.get("storage", {})
    if storage.get("camera_representation") != "deterministic_8x8_rgb_area_pool":
        errors.append("camera representation must match the frozen 8x8 feature schema")
    if config.get("training", {}).get("row_target_rule") != (
        "none_outside_exposure_otherwise_episode_mechanism"
    ):
        errors.append("training row target must not leak a future fault before exposure")
    if config.get("training", {}).get("final_test_access_before_lock") is not False:
        errors.append("final-test access must remain disabled before monitor lock")
    if config.get("primary_monitor_evaluation", {}).get("recovery_intervention") != "disabled":
        errors.append("recovery intervention must be disabled during monitor evaluation")
    policy = config.get("policy", {})
    if policy.get("frozen") is not True or policy.get("training_or_finetuning") is not False:
        errors.append("monitor data collection must keep the VLA frozen")
    integrity = config.get("formal_shard_integrity", {})
    required_integrity = {
        "schema_version": "monitor-shard-integrity-v1",
        "required_before_training": True,
        "require_exact_partition_seed_coverage": True,
        "require_three_conditions_per_seed": True,
        "require_cross_shard_seed_disjointness": True,
        "require_file_sha256": True,
        "require_clean_repository": True,
        "require_identical_provenance": True,
        "require_full_horizon": True,
    }
    if integrity != required_integrity:
        errors.append("formal_shard_integrity drifted from the frozen fail-closed gate")
    return errors


def validate_probe_protocol(
    probe: Mapping[str, Any], monitor_config: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if probe.get("protocol_version") != monitor_config.get("protocol_version"):
        errors.append("probe and monitor must target the same research protocol")
    if probe.get("probe_protocol_version") != "1.0":
        errors.append("probe_protocol_version must be 1.0")
    eligibility = probe.get("eligibility", {})
    for field in (
        "no_fault_label_access",
        "no_reward_or_success_access",
        "no_executed_action_access",
    ):
        if eligibility.get(field) is not True:
            errors.append(f"probe eligibility must enforce {field}")
    if int(eligibility.get("max_invocations_per_episode", -1)) != 1:
        errors.append("probe must permit exactly one bounded invocation per episode")
    declared = probe.get("probe", {})
    sequence = declared.get("action_sequence", [])
    maximum_steps = int(declared.get("max_environment_steps", -1))
    if maximum_steps <= 0 or len(sequence) != maximum_steps:
        errors.append("probe action sequence must exactly match max_environment_steps")
    if [row.get("step") for row in sequence] != list(range(maximum_steps)):
        errors.append("probe steps must be consecutive and zero-indexed")
    if probe.get("gates", {}).get("probe_must_not_run_before_monitor_calibration") is not True:
        errors.append("probe must remain blocked until monitor calibration")
    return errors


def factor_rows(config: Mapping[str, Any], partition: str) -> list[dict[str, Any]]:
    if partition not in PARTITIONS:
        raise ValueError(f"unsupported monitor partition: {partition}")
    errors = validate_monitor_protocol(config)
    if errors:
        raise ValueError(f"invalid monitor protocol: {errors}")
    seeds = [int(seed) for seed in config["splits"][SPLIT_KEYS[partition]]]
    fault = config["fault_sampling"]
    version = str(config["monitor_protocol_version"])
    assignments = {
        "onset_step": _balanced_assign(
            seeds, fault["onset_steps"], salt=f"{version}|{partition}|onset"
        ),
        "duration_steps": _balanced_assign(
            seeds, fault["duration_steps"], salt=f"{version}|{partition}|duration"
        ),
        "actuator_variant": _balanced_assign(
            seeds, fault["actuator_variants"], salt=f"{version}|{partition}|actuator"
        ),
        "observation_variant": _balanced_assign(
            seeds, fault["observation_variants"], salt=f"{version}|{partition}|observation"
        ),
        "camera_variant": _balanced_assign(
            seeds, fault["camera_variants"], salt=f"{version}|{partition}|camera"
        ),
    }
    return [
        {
            "seed": seed,
            "scene_seed_id": f"monitor-v1.0-{partition}-scene-{seed}",
            **{field: values[seed] for field, values in assignments.items()},
        }
        for seed in seeds
    ]


def _camera_keys(config: Mapping[str, Any], camera_variant: str) -> list[str]:
    mapping = config["collection_design"]["camera_variant_map"]
    if camera_variant not in mapping:
        raise ValueError(f"camera variant has no declared mapping: {camera_variant}")
    result = [str(key) for key in mapping[camera_variant]]
    if not result:
        raise ValueError(f"camera variant maps to no cameras: {camera_variant}")
    return result


def schedule_for_row(
    config: Mapping[str, Any], row: Mapping[str, Any], condition: str
) -> FaultSchedule:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported monitor condition: {condition}")
    if condition == "clean":
        return FaultSchedule(())
    seed = int(row["seed"])
    onset = int(row["onset_step"])
    duration = int(row["duration_steps"])
    if condition == "actuator_fault":
        variant = str(row["actuator_variant"])
        return FaultSchedule(
            [
                FaultSpec(
                    fault_id=f"monitor-v1.0-scene-{seed}-actuator",
                    kind="actuator_variant",
                    step=onset,
                    phase=FaultPhase.BEFORE_ACTION,
                    parameters={"duration": duration, "variant": variant},
                )
            ]
        )
    camera_variant = str(row["camera_variant"])
    return FaultSchedule(
        [
            FaultSpec(
                fault_id=f"monitor-v1.0-scene-{seed}-observation",
                kind="observation_variant",
                step=onset - 1,
                phase=FaultPhase.AFTER_STEP,
                parameters={
                    "duration": duration,
                    "variant": str(row["observation_variant"]),
                    "camera_keys": _camera_keys(config, camera_variant),
                    "first_affected_input_step": onset,
                },
            )
        ]
    )


def monitor_episode_plan(
    config: Mapping[str, Any], partition: str, *, seeds: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    rows = factor_rows(config, partition)
    if seeds is not None:
        selected = {int(seed) for seed in seeds}
        allowed = {int(row["seed"]) for row in rows}
        if not selected <= allowed:
            raise ValueError(
                f"requested seeds are outside {partition}: {sorted(selected - allowed)}"
            )
        rows = [row for row in rows if int(row["seed"]) in selected]
    plan: list[dict[str, Any]] = []
    for row in rows:
        for condition in CONDITIONS:
            episode_id = f"{row['scene_seed_id']}-{condition}"
            plan.append(
                {
                    "episode_id": episode_id,
                    "pair_id": row["scene_seed_id"],
                    "seed": int(row["seed"]),
                    "partition": partition,
                    "condition": condition,
                    "mechanism": "none" if condition == "clean" else condition,
                    "factor_row": dict(row),
                    "faults": schedule_for_row(config, row, condition).faults,
                }
            )
    version = str(config["monitor_protocol_version"])
    return sorted(
        plan,
        key=lambda item: hashlib.sha256(
            f"{version}|{partition}|{item['episode_id']}".encode()
        ).hexdigest(),
    )


def factor_balance(config: Mapping[str, Any], partition: str) -> dict[str, dict[str, int]]:
    rows = factor_rows(config, partition)
    fields = (
        "onset_step",
        "duration_steps",
        "actuator_variant",
        "observation_variant",
        "camera_variant",
    )
    result: dict[str, dict[str, int]] = {}
    for field in fields:
        counts = Counter(row[field] for row in rows)
        result[field] = {
            str(value): count
            for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
        }
    return result
