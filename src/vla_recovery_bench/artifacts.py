"""Immutable, self-describing experiment artifact helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .recording import to_jsonable

REQUIRED_RUN_ARTIFACTS = (
    "run_manifest.json",
    "episodes.jsonl",
    "metrics.json",
    "monitor_config.json",
    "calibration.json",
    "software_versions.json",
    "policy_state_before.json",
    "policy_state_after.json",
)


def ensure_empty_output_dir(path: str | Path) -> Path:
    """Create an output directory or fail closed if it already contains files."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    entries = tuple(output.iterdir())
    if entries:
        names = ", ".join(sorted(entry.name for entry in entries)[:8])
        suffix = "..." if len(entries) > 8 else ""
        raise FileExistsError(
            f"refusing to write into non-empty output directory: {output} ({names}{suffix})"
        )
    return output


def write_json_once(path: str | Path, value: Mapping[str, Any] | list[Any]) -> Path:
    """Write one JSON artifact without ever replacing an existing path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
    return target


def validate_required_artifacts(
    output_dir: str | Path,
    required: Iterable[str] = REQUIRED_RUN_ARTIFACTS,
) -> list[str]:
    """Return missing/empty required artifact names for a completed run."""
    output = Path(output_dir)
    errors: list[str] = []
    for name in required:
        path = output / name
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
        elif path.stat().st_size <= 0:
            errors.append(f"empty artifact: {path}")
    return errors


def validate_json_artifact_contract(output_dir: str | Path) -> list[str]:
    """Validate the minimum fields needed to score a completed run offline."""
    output = Path(output_dir)
    errors = validate_required_artifacts(output)
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        return errors
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid run_manifest.json: {error}")
        return errors
    if not isinstance(manifest, Mapping):
        errors.append("run_manifest.json must contain an object")
        return errors
    for field in ("protocol_version", "environment", "policy", "monitor", "seeds"):
        if field not in manifest:
            errors.append(f"run_manifest.json missing field: {field}")
    return errors
