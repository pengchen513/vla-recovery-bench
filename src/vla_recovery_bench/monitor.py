"""Auditable black-box monitor features and a small fault-conditioned model.

The monitor consumes only the fields declared by ``monitor_training_v1_0``:
observations, requested actions, chunk metadata, and declared latency.  Fault
labels, executed actions, rewards, and terminal information are deliberately
kept out of this module's online API.  The implementation is a reproducible
Phase-1 reference model, not a claim that this compact model is the final
paper architecture.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .groot_adapter import (
    ACTION_DIMS,
    ACTION_HORIZON,
    CAMERA_SHAPES,
    PROMPT_KEY,
    STATE_SHAPES,
    flatten_observation,
    task_description,
)
from .types import Action, MonitorContext, MonitorDecision, Observation

FEATURE_VERSION = "monitor-inputs-v1.0"
MECHANISMS = ("none", "actuator_fault", "observation_fault")
MECHANISM_TO_INDEX = {name: index for index, name in enumerate(MECHANISMS)}
POOL_SIZE = 8
WINDOW_SIZE = 16
INSTRUCTION_DIM = 16


def _numeric(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise ValueError(f"{name} must be numeric, got dtype={array.dtype}")
    array = array.astype(np.float32, copy=False).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _pool_rgb(value: Any, size: int = POOL_SIZE) -> np.ndarray:
    """Deterministically average an RGB image into a small model input."""
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected HxWx3/HxWx4 image, got {array.shape}")
    rgb = array[..., :3].astype(np.float32, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        rgb /= 255.0
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    height, width = rgb.shape[:2]
    rows = np.array_split(rgb, size, axis=0)
    pooled_rows = []
    for row in rows:
        bins = _column_bins(width, size)
        pooled_rows.append(
            np.stack(
                [row[:, start:stop].mean(axis=(0, 1)) for start, stop in bins], axis=0
            )
        )
    pooled = np.stack(pooled_rows, axis=0)
    return pooled.astype(np.float32, copy=False).reshape(-1)


def _column_bins(width: int, size: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, width, size + 1, dtype=np.int64)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(size)]


def _instruction_features(instruction: str) -> np.ndarray:
    if not instruction:
        raise ValueError("task instruction must not be empty")
    result = np.zeros(INSTRUCTION_DIM, dtype=np.float32)
    encoded = instruction.encode("utf-8")
    for index in range(max(1, len(encoded) - 2)):
        gram = encoded[index : index + 3]
        digest = hashlib.sha256(gram).digest()
        bucket = int.from_bytes(digest[:2], "big") % INSTRUCTION_DIM
        sign = 1.0 if digest[2] & 1 else -1.0
        result[bucket] += sign
    norm = float(np.linalg.norm(result))
    if norm:
        result /= norm
    return result


def validate_monitor_observation(observation: Observation) -> dict[str, Any]:
    """Validate every declared leaf without silently dropping a field."""
    flattened = flatten_observation(observation)
    expected = set(CAMERA_SHAPES) | set(STATE_SHAPES) | {PROMPT_KEY}
    if set(flattened) != expected:
        raise ValueError(
            "observation keys do not match the monitor contract: "
            f"missing={sorted(expected - set(flattened))}, "
            f"unexpected={sorted(set(flattened) - expected)}"
        )
    for key, shape in CAMERA_SHAPES.items():
        value = np.asarray(flattened[key])
        if value.shape != shape or value.dtype != np.uint8:
            raise ValueError(
                f"camera contract mismatch for {key}: expected {shape}/uint8, "
                f"got {value.shape}/{value.dtype}"
            )
    for key, shape in STATE_SHAPES.items():
        value = np.asarray(flattened[key])
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(
                f"state contract mismatch for {key}: expected {shape}/floating, "
                f"got {value.shape}/{value.dtype}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"state field contains NaN or Inf: {key}")
    task_description(observation)
    return flattened


def _structured_action(action: Action, *, name: str) -> np.ndarray:
    if not isinstance(action, Mapping):
        raise ValueError(f"{name} must be the structured RoboCasa action mapping")
    if set(action) != set(ACTION_DIMS):
        raise ValueError(
            f"{name} keys do not match the action contract: "
            f"missing={sorted(set(ACTION_DIMS) - set(action))}, "
            f"unexpected={sorted(set(action) - set(ACTION_DIMS))}"
        )
    values = []
    for key, width in ACTION_DIMS.items():
        array = _numeric(action[key], name=f"{name}.{key}")
        if array.shape != (width,):
            raise ValueError(f"{name}.{key} expected shape {(width,)}, got {array.shape}")
        values.append(array)
    return np.concatenate(values, axis=0).astype(np.float32, copy=False)


def flatten_action_chunk(action_chunk: Sequence[Action]) -> np.ndarray:
    if len(action_chunk) != ACTION_HORIZON:
        raise ValueError(
            f"requested action chunk must contain exactly {ACTION_HORIZON} actions; "
            f"got {len(action_chunk)}"
        )
    return np.stack(
        [_structured_action(action, name=f"requested_action_chunk[{index}]")
         for index, action in enumerate(action_chunk)],
        axis=0,
    )


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for camera in CAMERA_SHAPES:
        prefix = camera.replace(".", "_")
        names.extend(
            f"{prefix}.current_pool_{index}"
            for index in range(POOL_SIZE * POOL_SIZE * 3)
        )
        names.extend(
            f"{prefix}.delta_pool_{index}"
            for index in range(POOL_SIZE * POOL_SIZE * 3)
        )
    for key, shape in STATE_SHAPES.items():
        prefix = key.replace(".", "_")
        names.extend(f"{prefix}.current_{index}" for index in range(int(np.prod(shape))))
        names.extend(f"{prefix}.delta_{index}" for index in range(int(np.prod(shape))))
    names.extend(
        f"requested_action_chunk_{index}"
        for index in range(ACTION_HORIZON * sum(ACTION_DIMS.values()))
    )
    names.extend(("chunk_position", "chunk_length", "remaining_horizon", "policy_latency_ms"))
    names.extend(f"instruction_hash_{index}" for index in range(INSTRUCTION_DIM))
    return tuple(names)


FEATURE_NAMES = feature_names()


def compact_model_feature(feature: np.ndarray) -> np.ndarray:
    """Reduce the audited tensor into modality-preserving temporal statistics."""
    raw = np.asarray(feature, dtype=np.float32).reshape(-1)
    if raw.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"feature vector has unexpected shape {raw.shape}")
    values: list[np.ndarray] = []
    offset = 0
    camera_width = POOL_SIZE * POOL_SIZE * 3
    for _ in CAMERA_SHAPES:
        current = raw[offset : offset + camera_width]
        offset += camera_width
        delta = raw[offset : offset + camera_width]
        offset += camera_width
        values.append(
            np.asarray(
                [
                    current.mean(),
                    current.std(),
                    current.min(),
                    current.max(),
                    delta.mean(),
                    delta.std(),
                    delta.min(),
                    delta.max(),
                    np.mean(np.abs(delta)),
                    np.max(np.abs(delta)),
                ],
                dtype=np.float32,
            )
        )
    for shape in STATE_SHAPES.values():
        width = int(np.prod(shape))
        values.append(raw[offset : offset + 2 * width])
        offset += 2 * width
    action_width = sum(ACTION_DIMS.values())
    action_chunk = raw[offset : offset + ACTION_HORIZON * action_width].reshape(
        ACTION_HORIZON, action_width
    )
    offset += ACTION_HORIZON * action_width
    values.extend(
        (
            action_chunk.mean(axis=0),
            action_chunk.std(axis=0),
            action_chunk.min(axis=0),
            action_chunk.max(axis=0),
            action_chunk[0],
            action_chunk[-1],
        )
    )
    values.append(raw[offset : offset + 4])
    offset += 4
    values.append(raw[offset : offset + INSTRUCTION_DIM])
    offset += INSTRUCTION_DIM
    if offset != raw.size:
        raise RuntimeError(f"compact feature parser consumed {offset}/{raw.size} values")
    result = np.concatenate(values).astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError("compact monitor feature contains NaN or Inf")
    return result


MODEL_FEATURE_COUNT = len(compact_model_feature(np.zeros(len(FEATURE_NAMES), dtype=np.float32)))


def context_to_feature(context: MonitorContext) -> np.ndarray:
    """Convert one monitor context to a fixed, lossless-schema feature vector."""
    current = validate_monitor_observation(context.observation)
    previous = validate_monitor_observation(context.previous_observation)
    chunks = flatten_action_chunk(context.action_chunk)
    # Keep the explicit current requested action check even though it is the
    # first action in the complete chunk for the audited GR00T adapter.
    current_action = _structured_action(context.action, name="requested_action")
    position = context.chunk.position_in_chunk
    if not np.allclose(current_action, chunks[position], rtol=0.0, atol=1e-6):
        raise ValueError(
            "requested action must equal its declared position in requested_action_chunk"
        )
    values: list[np.ndarray] = []
    for key in CAMERA_SHAPES:
        now = _pool_rgb(current[key])
        old = _pool_rgb(previous[key])
        values.extend((now, now - old))
    for key in STATE_SHAPES:
        now = _numeric(current[key], name=key)
        old = _numeric(previous[key], name=f"previous.{key}")
        if now.shape != old.shape:
            raise ValueError(f"state shape changed between observations for {key}")
        values.extend((now, now - old))
    values.append(chunks.reshape(-1).astype(np.float32, copy=False))
    metadata = np.asarray(
        [
            context.chunk.position_in_chunk,
            context.chunk.chunk_length,
            context.chunk.remaining_horizon,
            float(context.chunk.policy_inference_latency_ms or 0.0),
        ],
        dtype=np.float32,
    )
    values.append(metadata)
    values.append(_instruction_features(context.instruction))
    result = np.concatenate(values).astype(np.float32, copy=False)
    if result.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(
            "feature schema implementation drift: "
            f"expected {len(FEATURE_NAMES)}, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("monitor feature vector contains NaN or Inf")
    return result


def context_to_feature_dict(context: MonitorContext) -> dict[str, float]:
    """Return a JSON-friendly feature map for audits and unit tests."""
    vector = context_to_feature(context)
    return {name: float(value) for name, value in zip(FEATURE_NAMES, vector, strict=True)}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(np.clip(shifted, -60.0, 60.0))
    return values / np.sum(values, axis=-1, keepdims=True)


def normalized_posterior_entropy(posterior: Mapping[str, float] | Sequence[float]) -> float:
    """Return Shannon entropy normalized to the three-class maximum.

    The monitor's mechanism posterior has three classes.  Keeping this helper
    independent of the simulator makes the probe trigger auditable and avoids
    silently changing the posterior representation at the runner boundary.
    """
    if isinstance(posterior, Mapping):
        if set(posterior) != set(MECHANISMS):
            raise ValueError(
                "posterior mapping must contain exactly the monitor mechanisms: "
                f"{MECHANISMS}"
            )
        try:
            values = np.asarray([posterior[name] for name in MECHANISMS], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("posterior mapping values must be numeric") from error
    else:
        values = np.asarray(posterior, dtype=np.float64).reshape(-1)
    if values.shape != (len(MECHANISMS),) or not np.all(np.isfinite(values)):
        raise ValueError("posterior must contain one finite probability per mechanism")
    if np.any(values < 0.0):
        raise ValueError("posterior probabilities must be non-negative")
    total = float(values.sum())
    if total <= 0.0 or not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"posterior probabilities must sum to one; got {total}")
    positive = values[values > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return float(np.clip(entropy / math.log(len(MECHANISMS)), 0.0, 1.0))


class FaultConditionedTemporalMonitor:
    """Deterministic temporal encoder plus a three-class mechanism head.

    Training is an offline supervised operation.  Online ``observe`` only sees
    ``MonitorContext`` and never accepts a label or simulator transition.
    """

    name = "fault_conditioned_temporal_reference_v1"

    def __init__(self, *, window_size: int = WINDOW_SIZE, seed: int = 14042026) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self.seed = int(seed)
        self._history: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.threshold_: float = 0.5
        self.calibration_: dict[str, Any] = {"status": "not_calibrated"}

    @property
    def fitted(self) -> bool:
        return all(
            value is not None for value in (self.mean_, self.scale_, self.weights_, self.bias_)
        )

    @property
    def embedding_dim(self) -> int:
        return MODEL_FEATURE_COUNT * 4

    def reset(self) -> None:
        self._history.clear()

    def _embed_one(self, feature: np.ndarray) -> np.ndarray:
        if feature.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"feature vector has unexpected shape {feature.shape}")
        self._history.append(compact_model_feature(feature))
        stack = np.stack(tuple(self._history), axis=0)
        first = stack[0]
        last = stack[-1]
        return np.concatenate((last, stack.mean(axis=0), stack.std(axis=0), last - first)).astype(
            np.float32, copy=False
        )

    def embed_sequence(self, features: np.ndarray, *, reset: bool = True) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (N, {len(FEATURE_NAMES)}), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("features contain NaN or Inf")
        if reset:
            self.reset()
        return np.stack([self._embed_one(row) for row in array], axis=0)

    def _embed_rows(
        self, features: np.ndarray, episode_ids: Sequence[str] | None
    ) -> np.ndarray:
        if episode_ids is None:
            return self.embed_sequence(features)
        if len(episode_ids) != len(features):
            raise ValueError("features and episode_ids length mismatch")
        rows: list[np.ndarray] = []
        previous_id: str | None = None
        for feature, episode_id in zip(features, episode_ids, strict=True):
            current_id = str(episode_id)
            if current_id != previous_id:
                self.reset()
                previous_id = current_id
            rows.append(self._embed_one(np.asarray(feature, dtype=np.float32)))
        return np.stack(rows, axis=0)

    def fit(
        self,
        features: np.ndarray,
        labels: Sequence[int] | np.ndarray,
        *,
        episode_ids: Sequence[str] | None = None,
        epochs: int = 80,
        learning_rate: float = 0.08,
        l2: float = 1e-4,
    ) -> dict[str, Any]:
        if epochs <= 0 or learning_rate <= 0 or l2 < 0:
            raise ValueError("epochs and learning_rate must be positive; l2 must be non-negative")
        feature_array = np.asarray(features, dtype=np.float32)
        embeddings = self._embed_rows(feature_array, episode_ids)
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        if target.shape[0] != embeddings.shape[0]:
            raise ValueError("features and labels length mismatch")
        if np.any((target < 0) | (target >= len(MECHANISMS))):
            raise ValueError(f"labels must be in [0, {len(MECHANISMS) - 1}]")
        self.mean_ = embeddings.mean(axis=0).astype(np.float32)
        self.scale_ = embeddings.std(axis=0).astype(np.float32)
        self.scale_[self.scale_ < 1e-6] = 1.0
        normalized = (embeddings - self.mean_) / self.scale_
        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(0.0, 0.01, size=(len(MECHANISMS), normalized.shape[1])).astype(
            np.float32
        )
        self.bias_ = np.zeros(len(MECHANISMS), dtype=np.float32)
        counts = np.bincount(target, minlength=len(MECHANISMS)).astype(np.float32)
        if np.any(counts == 0):
            raise ValueError(
                "training labels must contain all mechanisms; "
                f"counts={counts.tolist()}"
            )
        target_prior = np.asarray([0.5, 0.25, 0.25], dtype=np.float32)
        row_weights = target_prior[target] / counts[target]
        row_weights *= normalized.shape[0] / row_weights.sum()
        one_hot = np.eye(len(MECHANISMS), dtype=np.float32)[target]
        losses: list[float] = []
        for _ in range(epochs):
            logits = normalized @ self.weights_.T + self.bias_
            probabilities = _softmax(logits)
            weighted_error = (probabilities - one_hot) * row_weights[:, None]
            gradient_w = (weighted_error.T @ normalized) / normalized.shape[0]
            gradient_b = weighted_error.mean(axis=0)
            gradient_w += l2 * self.weights_
            self.weights_ -= learning_rate * gradient_w
            self.bias_ -= learning_rate * gradient_b
            losses.append(
                float(
                    -np.sum(
                        row_weights
                        * np.log(
                            np.clip(probabilities[np.arange(len(target)), target], 1e-8, 1.0)
                        )
                    )
                    / len(target)
                )
            )
        return {
            "status": "fitted",
            "rows": int(len(target)),
            "class_counts": {name: int(counts[index]) for index, name in enumerate(MECHANISMS)},
            "target_class_prior": {
                name: float(target_prior[index]) for index, name in enumerate(MECHANISMS)
            },
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "final_weighted_log_loss": losses[-1],
            "feature_version": FEATURE_VERSION,
            "window_size": self.window_size,
        }

    def _predict_embedding(self, embedding: np.ndarray) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("monitor must be fitted before prediction")
        assert self.mean_ is not None and self.scale_ is not None
        assert self.weights_ is not None and self.bias_ is not None
        normalized = (embedding - self.mean_) / self.scale_
        posterior = _softmax(normalized @ self.weights_.T + self.bias_)
        probabilities = posterior.reshape(-1)
        risk = float(1.0 - probabilities[MECHANISM_TO_INDEX["none"]])
        predicted_index = int(np.argmax(probabilities))
        entropy = normalized_posterior_entropy(probabilities)
        return {
            "risk": risk,
            "posterior": {
                name: float(probabilities[index]) for index, name in enumerate(MECHANISMS)
            },
            "predicted_mechanism": MECHANISMS[predicted_index],
            "failure_detected": bool(risk >= self.threshold_),
            "threshold": float(self.threshold_),
            "normalized_entropy": entropy,
        }

    def predict_features(self, feature: np.ndarray) -> dict[str, Any]:
        embedding = self._embed_one(np.asarray(feature, dtype=np.float32))
        return self._predict_embedding(embedding)

    def observe(self, context: MonitorContext) -> MonitorDecision:
        result = self.predict_features(context_to_feature(context))
        predicted = result["predicted_mechanism"]
        failure_type = predicted if predicted != "none" and result["failure_detected"] else None
        return MonitorDecision(
            failure_detected=bool(result["failure_detected"]),
            confidence=float(result["risk"]),
            failure_type=failure_type,
            evidence={
                "risk": result["risk"],
                "posterior": result["posterior"],
                "threshold": result["threshold"],
                "feature_version": FEATURE_VERSION,
                "predicted_mechanism": predicted,
                "normalized_entropy": result["normalized_entropy"],
            },
        )

    def calibrate_clean_episode_maxima(
        self,
        clean_episode_features: Sequence[np.ndarray],
        *,
        false_intervention_rate: float = 0.05,
    ) -> dict[str, Any]:
        if not 0.0 < false_intervention_rate < 1.0:
            raise ValueError("false_intervention_rate must be in (0, 1)")
        maxima: list[float] = []
        for features in clean_episode_features:
            self.reset()
            predictions = [self.predict_features(row)["risk"] for row in np.asarray(features)]
            if predictions:
                maxima.append(float(max(predictions)))
        if not maxima:
            raise ValueError("at least one non-empty clean calibration episode is required")
        ordered = np.sort(np.asarray(maxima, dtype=np.float64))
        rank = int(math.ceil((len(ordered) + 1) * (1.0 - false_intervention_rate)))
        index = min(rank, len(ordered)) - 1
        # Alarms use >=, so move one representable value above the selected
        # conformal order statistic. If rank > n this yields zero calibration
        # alarms, which is the only finite-sample choice compatible with alpha.
        self.threshold_ = float(np.nextafter(ordered[index], np.inf))
        alarms = sum(value >= self.threshold_ for value in maxima)
        self.calibration_ = {
            "status": "calibrated",
            "method": "split_conformal_clean_episode_maximum",
            "target_false_intervention_rate": false_intervention_rate,
            "threshold": self.threshold_,
            "clean_episode_count": len(maxima),
            "conformal_rank": rank,
            "clean_alarm_episodes_at_threshold": int(alarms),
            "empirical_clean_alarm_rate": float(alarms / len(maxima)),
            "sequential_dependence_note": (
                "episode maximum is the calibration unit; steps are not treated as independent"
            ),
        }
        return dict(self.calibration_)

    def save(self, path: str | Path) -> Path:
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted monitor")
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite monitor checkpoint: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        assert self.mean_ is not None and self.scale_ is not None
        assert self.weights_ is not None and self.bias_ is not None
        np.savez(
            target,
            mean=self.mean_,
            scale=self.scale_,
            weights=self.weights_,
            bias=self.bias_,
            threshold=np.asarray([self.threshold_], dtype=np.float64),
            window_size=np.asarray([self.window_size], dtype=np.int64),
            seed=np.asarray([self.seed], dtype=np.int64),
            metadata=np.asarray(
                json.dumps(
                    {
                        "name": self.name,
                        "feature_version": FEATURE_VERSION,
                        "feature_count": len(FEATURE_NAMES),
                        "model_feature_count": MODEL_FEATURE_COUNT,
                        "mechanisms": MECHANISMS,
                    },
                    sort_keys=True,
                )
            ),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> FaultConditionedTemporalMonitor:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("feature_version") != FEATURE_VERSION:
                raise ValueError("monitor feature version mismatch")
            if tuple(metadata.get("mechanisms", ())) != MECHANISMS:
                raise ValueError("monitor mechanism class mismatch")
            monitor = cls(
                window_size=int(data["window_size"][0]),
                seed=int(data["seed"][0]),
            )
            monitor.mean_ = np.asarray(data["mean"], dtype=np.float32)
            monitor.scale_ = np.asarray(data["scale"], dtype=np.float32)
            monitor.weights_ = np.asarray(data["weights"], dtype=np.float32)
            monitor.bias_ = np.asarray(data["bias"], dtype=np.float32)
            monitor.threshold_ = float(data["threshold"][0])
        return monitor


def monitor_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mechanism_index(name: str) -> int:
    try:
        return MECHANISM_TO_INDEX[name]
    except KeyError as error:
        raise ValueError(f"unsupported monitor mechanism: {name}") from error
