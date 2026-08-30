"""Pure v1.4 pilot schedule derivation and contract checks."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import validate_json_artifact_contract
from .faults import FaultSchedule
from .types import FaultPhase, FaultSpec

CONDITIONS = ("clean", "actuator_fault", "observation_fault")


def derive_pilot_schedule(row: Mapping[str, Any], condition: str) -> FaultSchedule:
    """Derive one immutable episode schedule from a v1.4 factor row."""
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported pilot condition: {condition}")
    seed = int(row["seed"])
    onset = int(row["onset_step"])
    duration = int(row["duration_steps"])
    if onset < 1 or duration <= 0:
        raise ValueError("pilot onset must be positive and duration must be positive")
    if condition == "clean":
        return FaultSchedule(())
    if condition == "actuator_fault":
        variant = str(row["actuator_variant"])
        return FaultSchedule(
            [
                FaultSpec(
                    fault_id=f"pilot-v1.4-scene-{seed}-actuator",
                    kind="actuator_variant",
                    step=onset,
                    phase=FaultPhase.BEFORE_ACTION,
                    parameters={"duration": duration, "variant": variant},
                )
            ]
        )
    camera = str(row["observation_camera"])
    variant = str(row["observation_variant"])
    return FaultSchedule(
        [
            FaultSpec(
                fault_id=f"pilot-v1.4-scene-{seed}-observation",
                kind="observation_variant",
                step=onset - 1,
                phase=FaultPhase.AFTER_STEP,
                parameters={
                    "duration": duration,
                    "variant": variant,
                    "camera_key": camera,
                    "first_affected_input_step": onset,
                },
            )
        ]
    )


def pilot_episode_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the crossed scene rows into clean and two fault conditions."""
    rows = config.get("scene_seed_factors", [])
    if not isinstance(rows, list):
        raise TypeError("scene_seed_factors must be a list")
    plan: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each scene_seed_factors entry must be an object")
        seed = int(row["seed"])
        scene_id = str(row.get("scene_seed_id", f"pilot-v1.4-scene-{seed}"))
        for condition in CONDITIONS:
            schedule = derive_pilot_schedule(row, condition)
            plan.append(
                {
                    "episode_id": f"{scene_id}-{condition}",
                    "pair_id": scene_id,
                    "seed": seed,
                    "condition": condition,
                    "faults": schedule.faults,
                }
            )
    return plan


def validate_pilot_artifacts(
    output_dir: str | Path,
    *,
    expected_episode_count: int,
) -> list[str]:
    """Validate a completed pilot without exposing audit labels online."""
    output = Path(output_dir)
    errors = validate_json_artifact_contract(output)

    def load_json(name: str) -> dict[str, Any] | None:
        path = output / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid {path}: {error}")
            return None
        if not isinstance(value, dict):
            errors.append(f"{path} must contain a JSON object")
            return None
        return value

    def load_jsonl(name: str) -> list[dict[str, Any]]:
        path = output / name
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing or empty pilot stream: {path}")
            return []
        records: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        errors.append(f"{path}:{line_number} must contain a JSON object")
                        continue
                    records.append(value)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid {path}: {error}")
        return records

    episodes = load_jsonl("episodes.jsonl")
    monitor = load_jsonl("monitor_stream.jsonl")

    # The v1.4 diagnostic runner deliberately renamed the privileged channel
    # to make its separation from the online monitor channel explicit.  Keep
    # the old name as a backwards-compatible input for the original pilot
    # runner, but never require both files or silently treat one schema as the
    # other.
    canonical_audit = output / "privileged_audit.jsonl"
    if canonical_audit.is_file():
        audit_name = "privileged_audit.jsonl"
    else:
        audit_name = "audit_stream.jsonl"
    audit = load_jsonl(audit_name)
    if len(episodes) != expected_episode_count:
        errors.append(
            f"episodes.jsonl has {len(episodes)} records; expected {expected_episode_count}"
        )
    # A diagnostic pair intentionally has the same episode_id in its passive
    # and probe arms.  Uniqueness is therefore (episode_id, arm) for the v1.4
    # schema, while legacy single-arm pilots retain episode_id uniqueness.
    has_arm = any("arm" in record for record in episodes)
    if has_arm:
        keys = [(str(record.get("episode_id")), str(record.get("arm", ""))) for record in episodes]
        if any(not arm for _, arm in keys):
            errors.append("dual-arm episodes.jsonl contains a record without arm")
        if len(set(keys)) != len(keys):
            errors.append("episodes.jsonl contains duplicate episode_id/arm pairs")
        pair_arms: dict[str, set[str]] = defaultdict(set)
        for record in episodes:
            pair_arms[str(record.get("pair_id", record.get("episode_id")))].add(
                str(record.get("arm", ""))
            )
        for pair_id, arms in pair_arms.items():
            if arms != {"passive_only", "passive_plus_probe"}:
                errors.append(
                    f"pair {pair_id} does not contain both diagnostic arms: {sorted(arms)}"
                )
    else:
        episode_ids = [str(record.get("episode_id")) for record in episodes]
        if len(set(episode_ids)) != len(episode_ids):
            errors.append("episodes.jsonl contains duplicate episode_id values")
    conditions = [record.get("condition") for record in episodes]
    expected_per_condition = expected_episode_count // len(CONDITIONS)
    for condition in CONDITIONS:
        if conditions.count(condition) != expected_per_condition:
            errors.append(
                f"condition {condition} has {conditions.count(condition)} episodes; "
                f"expected {expected_per_condition}"
            )

    forbidden_top_level = {
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
    for index, record in enumerate(monitor):
        leaked = forbidden_top_level & set(record)
        if leaked:
            errors.append(f"monitor_stream record {index} leaks fields: {sorted(leaked)}")
            break
        if "episode_token" not in record:
            errors.append(f"monitor_stream record {index} is missing episode_token")
            break
        chunk = record.get("chunk")
        action_chunk = record.get("requested_action_chunk")
        if not isinstance(chunk, Mapping) or not isinstance(action_chunk, list):
            errors.append(
                f"monitor_stream record {index} is missing the complete requested action chunk"
            )
            break
        expected_chunk_length = chunk.get("chunk_length")
        if not isinstance(expected_chunk_length, int) or len(action_chunk) != expected_chunk_length:
            errors.append(
                f"monitor_stream record {index} has an incomplete requested action chunk: "
                f"expected={expected_chunk_length}, got={len(action_chunk)}"
            )
            break
    if not audit:
        errors.append(f"{audit_name} contains no records")

    before = load_json("policy_state_before.json")
    after = load_json("policy_state_after.json")
    if before is not None and after is not None:
        before_hash = before.get("current_parameter_sha256")
        after_hash = after.get("current_parameter_sha256")
        if not before_hash or before_hash != after_hash:
            errors.append("policy parameter hash changed or is missing")
        if before.get("model_training") or after.get("model_training"):
            errors.append("policy was in training mode")
        if not before.get("all_parameters_frozen") or not after.get("all_parameters_frozen"):
            errors.append("policy did not remain fully frozen")

    metrics = load_json("metrics.json")
    manifest = load_json("run_manifest.json")
    if metrics is not None:
        if metrics.get("episode_count") != expected_episode_count:
            errors.append("metrics episode_count does not match the run plan")
        if metrics.get("status") != "completed":
            errors.append("metrics status is not completed")
    if manifest is not None:
        plan = manifest.get("episode_plan", [])
        if not isinstance(plan, list) or len(plan) != expected_episode_count:
            errors.append("run_manifest episode_plan does not match expected episode count")
    return errors
