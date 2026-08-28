"""Separated monitor-input and privileged-label dataset artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .monitor import FEATURE_NAMES, FEATURE_VERSION, MECHANISMS

FORBIDDEN_INPUT_METADATA = frozenset(
    {
        "condition",
        "executed_action",
        "exposure",
        "fault",
        "fault_duration",
        "fault_onset",
        "fault_schedule",
        "info",
        "mechanism",
        "reward",
        "seed",
        "success",
        "terminated",
        "truncated",
    }
)


@dataclass(frozen=True)
class MonitorDatasetEpisode:
    token: str
    features: np.ndarray
    control_steps: np.ndarray
    observation_steps: np.ndarray
    instruction: str
    mechanism: str
    exposure: np.ndarray
    seed: int
    pair_id: str
    success: bool
    partition: str = ""
    condition: str = ""
    factor_row: dict[str, Any] | None = None
    fault_schedule: list[Any] | None = None
    episode_id: str = ""


def _attribute_text(value: Any) -> str:
    """Decode h5py string attributes consistently across h5py versions."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _json_attribute(group: Any, key: str, *, default: Any = None) -> Any:
    name = f"{key}_json"
    if name not in group.attrs:
        return default
    try:
        return json.loads(_attribute_text(group.attrs[name]))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON label attribute {name}") from error


class MonitorDatasetWriter:
    """Write independent HDF5 channels without cross-channel attributes."""

    def __init__(self, output_dir: str | Path, *, partition: str, protocol_sha256: str) -> None:
        import h5py

        output = Path(output_dir)
        self.input_path = output / "monitor_inputs.h5"
        self.label_path = output / "offline_labels.h5"
        if self.input_path.exists() or self.label_path.exists():
            raise FileExistsError("refusing to overwrite an existing monitor dataset channel")
        self._inputs = h5py.File(self.input_path, "x")
        self._labels = h5py.File(self.label_path, "x")
        self._inputs.attrs["feature_version"] = FEATURE_VERSION
        self._inputs.attrs["feature_names_json"] = json.dumps(FEATURE_NAMES)
        self._inputs.attrs["partition"] = partition
        self._inputs.attrs["protocol_sha256"] = protocol_sha256
        self._inputs.attrs["contains_privileged_labels"] = False
        self._labels.attrs["partition"] = partition
        self._labels.attrs["protocol_sha256"] = protocol_sha256
        self._labels.attrs["contains_privileged_labels"] = True
        self._inputs.create_group("episodes")
        self._labels.create_group("episodes")

    def write_episode(
        self,
        *,
        token: str,
        features: np.ndarray,
        control_steps: np.ndarray,
        observation_steps: np.ndarray,
        instruction: str,
        label: dict[str, Any],
        exposure: np.ndarray,
    ) -> None:
        if not token or "/" in token:
            raise ValueError("episode token must be a non-empty HDF5-safe identifier")
        if token in self._inputs["episodes"] or token in self._labels["episodes"]:
            raise ValueError(f"duplicate episode token: {token}")
        feature_array = np.asarray(features, dtype=np.float32)
        control = np.asarray(control_steps, dtype=np.int32).reshape(-1)
        observation = np.asarray(observation_steps, dtype=np.int32).reshape(-1)
        exposed = np.asarray(exposure, dtype=np.bool_).reshape(-1)
        if feature_array.ndim != 2 or feature_array.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (N, {len(FEATURE_NAMES)}), got {feature_array.shape}"
            )
        row_count = feature_array.shape[0]
        if not all(len(values) == row_count for values in (control, observation, exposed)):
            raise ValueError("feature, step, and exposure arrays must have equal length")
        if not np.all(np.isfinite(feature_array)):
            raise ValueError("monitor features contain NaN or Inf")
        if label.get("mechanism") not in MECHANISMS:
            raise ValueError(f"unsupported mechanism label: {label.get('mechanism')}")

        inputs = self._inputs["episodes"].create_group(token)
        inputs.attrs["episode_token"] = token
        inputs.attrs["instruction"] = instruction
        inputs.create_dataset(
            "features",
            data=feature_array,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(max(row_count, 1), 64), feature_array.shape[1]),
        )
        inputs.create_dataset("control_steps", data=control)
        inputs.create_dataset("observation_steps", data=observation)

        labels = self._labels["episodes"].create_group(token)
        labels.attrs["episode_token"] = token
        for key, value in label.items():
            if isinstance(value, (dict, list, tuple)):
                labels.attrs[f"{key}_json"] = json.dumps(value, sort_keys=True)
            elif value is None:
                labels.attrs[f"{key}_is_none"] = True
            else:
                labels.attrs[key] = value
        labels.create_dataset("exposure", data=exposed)
        self._inputs.flush()
        self._labels.flush()

    def close(self) -> None:
        if self._inputs:
            self._inputs.close()
        if self._labels:
            self._labels.close()

    def __enter__(self) -> MonitorDatasetWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_monitor_dataset(
    output_dir: str | Path, *, expected_partition: str | None = None
) -> list[MonitorDatasetEpisode]:
    import h5py

    output = Path(output_dir)
    episodes: list[MonitorDatasetEpisode] = []
    with (
        h5py.File(output / "monitor_inputs.h5", "r") as inputs,
        h5py.File(output / "offline_labels.h5", "r") as labels,
    ):
        input_partition = _attribute_text(inputs.attrs["partition"])
        label_partition = _attribute_text(labels.attrs["partition"])
        if input_partition != label_partition:
            raise ValueError("monitor input and label partitions disagree")
        if expected_partition is not None and input_partition != expected_partition:
            raise ValueError(
                f"dataset partition mismatch: expected={expected_partition}, got={input_partition}"
            )
        if _attribute_text(inputs.attrs["feature_version"]) != FEATURE_VERSION:
            raise ValueError("monitor dataset feature version mismatch")
        if bool(inputs.attrs.get("contains_privileged_labels")):
            raise ValueError("monitor input channel is not marked label-free")
        if not bool(labels.attrs.get("contains_privileged_labels")):
            raise ValueError("offline label channel is not marked privileged")
        input_protocol = _attribute_text(inputs.attrs.get("protocol_sha256", ""))
        label_protocol = _attribute_text(labels.attrs.get("protocol_sha256", ""))
        if not input_protocol or input_protocol != label_protocol:
            raise ValueError("monitor input and label protocol hashes disagree")
        declared_names = tuple(
            json.loads(_attribute_text(inputs.attrs["feature_names_json"]))
        )
        if declared_names != FEATURE_NAMES:
            raise ValueError("monitor dataset feature schema mismatch")
        input_tokens = set(inputs["episodes"])
        label_tokens = set(labels["episodes"])
        if input_tokens != label_tokens:
            raise ValueError("monitor input and offline-label episode tokens disagree")
        for token in sorted(input_tokens):
            input_group = inputs["episodes"][token]
            label_group = labels["episodes"][token]
            attributes = {str(key).lower() for key in input_group.attrs}
            leaked = attributes & FORBIDDEN_INPUT_METADATA
            if leaked:
                raise ValueError(f"monitor input group {token} leaks metadata: {sorted(leaked)}")
            features = np.asarray(input_group["features"], dtype=np.float32)
            control_steps = np.asarray(input_group["control_steps"], dtype=np.int32)
            observation_steps = np.asarray(
                input_group["observation_steps"], dtype=np.int32
            )
            exposure = np.asarray(label_group["exposure"], dtype=np.bool_)
            if features.ndim != 2:
                raise ValueError(
                    f"episode {token} features must be a matrix, got {features.shape}"
                )
            if control_steps.ndim != 1 or observation_steps.ndim != 1 or exposure.ndim != 1:
                raise ValueError(f"episode {token} step and exposure datasets must be vectors")
            if not all(
                values.shape[0] == features.shape[0]
                for values in (control_steps, observation_steps, exposure)
            ):
                raise ValueError(f"episode {token} feature and step lengths disagree")
            mechanism = str(label_group.attrs["mechanism"])
            if mechanism not in MECHANISMS:
                raise ValueError(f"episode {token} has unsupported mechanism {mechanism}")
            episodes.append(
                MonitorDatasetEpisode(
                    token=token,
                    features=features,
                    control_steps=control_steps,
                    observation_steps=observation_steps,
                    instruction=str(input_group.attrs["instruction"]),
                    mechanism=mechanism,
                    exposure=exposure,
                    seed=int(label_group.attrs["seed"]),
                    pair_id=_attribute_text(label_group.attrs["pair_id"]),
                    success=bool(label_group.attrs["success"]),
                    partition=_attribute_text(
                        label_group.attrs.get("partition", label_partition)
                    ),
                    condition=_attribute_text(label_group.attrs["condition"]),
                    factor_row=_json_attribute(label_group, "factor_row"),
                    fault_schedule=_json_attribute(label_group, "fault_schedule"),
                    episode_id=_attribute_text(label_group.attrs.get("episode_id", "")),
                )
            )
    return episodes


def validate_monitor_dataset(
    output_dir: str | Path,
    *,
    expected_partition: str,
    expected_episode_count: int,
) -> list[str]:
    output = Path(output_dir)
    errors: list[str] = []
    required = (
        "monitor_inputs.h5",
        "offline_labels.h5",
        "dataset_index.jsonl",
        "episodes.jsonl",
        "audit_stream.jsonl",
        "policy_state_before.json",
        "policy_state_after.json",
    )
    for name in required:
        path = output / name
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing or empty monitor dataset artifact: {path}")
    if errors:
        return errors
    try:
        episodes = load_monitor_dataset(output, expected_partition=expected_partition)
    except (OSError, KeyError, TypeError, ValueError) as error:
        return [f"invalid monitor dataset: {error}"]
    if len(episodes) != expected_episode_count:
        errors.append(
            f"monitor dataset has {len(episodes)} episodes; expected {expected_episode_count}"
        )
    if len({episode.token for episode in episodes}) != len(episodes):
        errors.append("monitor dataset contains duplicate episode tokens")
    for episode in episodes:
        if episode.features.shape[0] <= 0:
            errors.append(f"episode {episode.token} has no monitor rows")
            break
        if not np.all(np.isfinite(episode.features)):
            errors.append(f"episode {episode.token} contains non-finite features")
            break
    try:
        before = json.loads((output / "policy_state_before.json").read_text())
        after = json.loads((output / "policy_state_after.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid policy state artifact: {error}")
    else:
        if before.get("current_parameter_sha256") != after.get("current_parameter_sha256"):
            errors.append("frozen policy parameter hash changed during monitor data collection")
        if before.get("model_training") or after.get("model_training"):
            errors.append("frozen policy entered training mode during monitor data collection")
        if not before.get("all_parameters_frozen") or not after.get("all_parameters_frozen"):
            errors.append("policy parameters were not fully frozen")
    return errors
