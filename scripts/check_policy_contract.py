#!/usr/bin/env python3
"""Check a frozen policy contract manifest against a RoboCasa probe report.

The manifest is intentionally metadata-only. This gate does not download or
load model weights; it prevents an unverified checkpoint from entering an
experiment until its interface is explicitly documented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "embodiment",
    "camera_keys",
    "image_shape",
    "image_preprocessing",
    "proprioception_keys",
    "state_normalization",
    "action_space",
    "action_keys",
    "action_dim",
    "model_action_dim",
    "action_horizon",
    "control_mode",
    "action_normalization",
    "prompt_key",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_files",
    "published_clean_baseline",
)


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_set_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(f"{record['sha256']}  {record['path']}\n".encode())
    return digest.hexdigest()


def check_manifest(manifest: dict[str, Any], probe: dict[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}"
        for field in REQUIRED_FIELDS
        if field not in manifest
    ]
    if errors:
        return errors
    observation = probe.get("observation", {})
    if not isinstance(observation, dict):
        return ["probe.observation must be an object"]
    actual_records = observation.get("contract", [])
    if not isinstance(actual_records, list) or any(
        not isinstance(record, dict) or "key" not in record for record in actual_records
    ):
        return ["probe.observation.contract must be a list of keyed objects"]
    image_records = observation.get("images", [])
    if not isinstance(image_records, list) or any(
        not isinstance(record, dict) or "key" not in record for record in image_records
    ):
        return ["probe.observation.images must be a list of keyed objects"]
    actual_by_key = {str(record["key"]): record for record in actual_records}
    actual_images = [str(image["key"]) for image in image_records]
    expected_cameras = list(manifest["camera_keys"])
    if actual_images != expected_cameras:
        errors.append(f"camera_keys mismatch: expected {expected_cameras}, got {actual_images}")
    expected_shape = list(manifest["image_shape"])
    for key in expected_cameras:
        if key not in actual_by_key or actual_by_key[key].get("shape") != expected_shape:
            errors.append(f"image contract mismatch for {key}")
    expected_state = list(manifest["proprioception_keys"])
    state_keys = [key for key in actual_by_key if key.startswith("state.")]
    if set(state_keys) != set(expected_state) or len(state_keys) != len(expected_state):
        errors.append(f"proprioception_keys mismatch: expected {expected_state}, got {state_keys}")
    expected_action = manifest["action_space"]
    actual_action = probe.get("action", {}).get("space", {})
    if expected_action != actual_action:
        errors.append("action_space metadata does not match RoboCasa probe")
    preprocessing = manifest["image_preprocessing"]
    if not isinstance(preprocessing, dict):
        errors.append("image_preprocessing must be an object")
    else:
        if preprocessing.get("source_shape") != expected_shape:
            errors.append("image_preprocessing.source_shape must match image_shape")
        if not isinstance(preprocessing.get("resize"), list) or not preprocessing["resize"]:
            errors.append("image_preprocessing.resize must be a non-empty list")
    if not isinstance(manifest["state_normalization"], dict) or not manifest[
        "state_normalization"
    ]:
        errors.append("state_normalization must be a non-empty object")
    action_spaces = expected_action.get("spaces") if isinstance(expected_action, dict) else None
    if not isinstance(action_spaces, dict):
        errors.append("action_space.spaces must be an object")
    else:
        expected_action_keys = list(manifest["action_keys"])
        if set(expected_action_keys) != set(action_spaces):
            errors.append("action_keys must exactly match action_space.spaces")
        try:
            inferred_action_dim = sum(
                math.prod(int(dimension) for dimension in space["shape"])
                for space in action_spaces.values()
            )
        except (KeyError, TypeError, ValueError):
            errors.append("action_space.spaces must declare numeric shapes")
        else:
            if int(manifest["action_dim"]) != inferred_action_dim:
                errors.append("action_dim does not match action_space.spaces")
            if int(manifest["model_action_dim"]) < inferred_action_dim:
                errors.append("model_action_dim must be at least action_dim")
    try:
        if int(manifest["action_horizon"]) <= 0:
            errors.append("action_horizon must be positive")
    except (TypeError, ValueError):
        errors.append("action_horizon must be an integer")
    if manifest["prompt_key"] not in actual_by_key:
        errors.append("prompt_key is absent from the RoboCasa probe")
    checkpoint_path = Path(str(manifest["checkpoint_path"]))
    records = manifest["checkpoint_files"]
    if not checkpoint_path.is_dir():
        errors.append(f"checkpoint directory does not exist: {checkpoint_path}")
    elif not isinstance(records, list) or not records:
        errors.append("checkpoint_files must be a non-empty list")
    else:
        for record in records:
            path = checkpoint_path / str(record["path"])
            if not path.is_file():
                errors.append(f"checkpoint file does not exist: {path}")
                continue
            if path.stat().st_size != int(record["size"]):
                errors.append(f"checkpoint file size does not match: {path}")
                continue
            if _sha256(path) != str(record["sha256"]):
                errors.append(f"checkpoint file SHA256 does not match: {path}")
        if _checkpoint_set_sha256(records) != str(manifest["checkpoint_sha256"]):
            errors.append("checkpoint_sha256 does not match checkpoint_files")
    if not isinstance(manifest["published_clean_baseline"], dict):
        errors.append("published_clean_baseline must be a metadata object")
    elif not manifest["published_clean_baseline"]:
        errors.append("published_clean_baseline must not be empty")
    for field in ("embodiment", "control_mode", "action_normalization", "prompt_key"):
        if not manifest[field]:
            errors.append(f"{field} must not be empty")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--probe", default="/home/pc/VLA/outputs/robocasa_contract.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = _load(args.manifest)
        probe = _load(args.probe)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(json.dumps({"status": "blocked", "errors": [str(error)]}, indent=2))
        return 2
    errors = check_manifest(manifest, probe)
    if errors:
        print(json.dumps({"status": "blocked", "errors": errors}, indent=2))
        return 2
    print(json.dumps({"status": "passed", "manifest": args.manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
