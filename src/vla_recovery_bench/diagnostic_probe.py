"""Pure helpers for the v1.1 diagnostic-probe protocol.

The module does not instantiate RoboCasa or a policy service. It is
intentionally split from the runner so threshold locking, online-firewall
checks, and offline statistics can be tested without a GPU or a simulator.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import Any

import numpy as np

from .monitor import MECHANISMS, normalized_posterior_entropy

PROBE_PROTOCOL_VERSION = "1.1"
MAX_CLEAN_UNION_RATE = 0.05
MAX_PROBE_STEPS = 4

# These names are checked recursively on serialized online events.  The
# controller may see posterior-derived diagnostics, but never simulator labels
# or the action that actually reached the robot.
FORBIDDEN_ONLINE_FIELDS = frozenset(
    {
        "arm",
        "condition",
        "executed_action",
        "fault",
        "fault_id",
        "fault_kind",
        "fault_schedule",
        "fault_type",
        "info",
        "mujoco_state",
        "pair_id",
        "reward",
        "seed",
        "success",
        "terminated",
        "truncated",
    }
)


def _key_components(key: Any) -> set[str]:
    """Normalize dotted/path-like field names for the online firewall."""
    return {
        component.lower().replace("-", "_")
        for component in str(key).replace("/", ".").split(".")
        if component
    }


def assert_online_event_safe(value: Any, path: str = "event") -> None:
    """Fail closed if an online event contains a privileged field name."""
    if is_dataclass(value) and not isinstance(value, type):
        assert_online_event_safe(vars(value), path)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            components = _key_components(key)
            if components & FORBIDDEN_ONLINE_FIELDS:
                raise ValueError(f"forbidden online probe field at {path}.{key}")
            assert_online_event_safe(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_online_event_safe(child, f"{path}[{index}]")


def monitor_parameter_sha256(monitor: Any) -> str:
    """Hash fitted monitor parameters, excluding mutable temporal history."""
    digest = hashlib.sha256()
    for name in ("mean_", "scale_", "weights_", "bias_"):
        value = getattr(monitor, name, None)
        if value is None:
            raise ValueError(f"monitor is not fitted: missing {name}")
        array = np.asarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(str(float(monitor.threshold_)).encode("ascii"))
    return digest.hexdigest()


def choose_entropy_threshold(
    clean_episode_summaries: Sequence[Mapping[str, Any]],
    *,
    risk_threshold: float,
    max_union_rate: float = MAX_CLEAN_UNION_RATE,
) -> dict[str, Any]:
    """Choose a deterministic entropy threshold under an episode union budget.

    ``clean_episode_summaries`` must contain one row per clean episode with
    ``maximum_risk`` and ``maximum_entropy``.  Risk alarms consume the budget
    first; the entropy threshold is then selected from the remaining episodes.
    The one-sided ``nextafter`` is required because the trigger uses ``>=``.
    """
    if not clean_episode_summaries:
        raise ValueError("at least one clean episode is required")
    if not math.isfinite(max_union_rate) or not 0.0 < max_union_rate < 1.0:
        raise ValueError("max_union_rate must be in (0, 1)")
    n = len(clean_episode_summaries)
    allowed = int(math.floor(max_union_rate * n + 1e-12))
    if allowed < 0:
        raise ValueError("union budget must be non-negative")
    if not math.isfinite(float(risk_threshold)):
        raise ValueError("risk_threshold must be finite")
    risk_values: list[float] = []
    entropy_values: list[float] = []
    for row in clean_episode_summaries:
        risk = float(row["maximum_risk"])
        entropy = float(row["maximum_entropy"])
        if not math.isfinite(risk) or not math.isfinite(entropy):
            raise ValueError("clean episode maxima must be finite")
        if not 0.0 <= risk <= 1.0:
            raise ValueError(f"maximum_risk must be in [0, 1], got {risk}")
        if not 0.0 <= entropy <= 1.0:
            raise ValueError(f"maximum_entropy must be in [0, 1], got {entropy}")
        risk_values.append(risk)
        entropy_values.append(entropy)
    risk_triggered = [risk >= risk_threshold for risk in risk_values]
    risk_count = sum(risk_triggered)
    if risk_count > allowed:
        raise ValueError(
            "risk alarm alone exceeds the clean joint-trigger budget: "
            f"risk_episodes={risk_count}, allowed={allowed}, total={n}"
        )
    nonrisk_entropy = sorted(
        entropy
        for entropy, risk in zip(entropy_values, risk_triggered, strict=True)
        if not risk
    )
    capacity = allowed - risk_count
    if capacity >= len(nonrisk_entropy):
        entropy_threshold = 0.0
        threshold_rule = "all_nonrisk_episodes_fit_remaining_budget"
    elif not nonrisk_entropy:
        entropy_threshold = float(np.nextafter(1.0, np.inf))
        threshold_rule = "no_nonrisk_episodes"
    elif capacity == 0:
        entropy_threshold = float(np.nextafter(max(nonrisk_entropy), np.inf))
        threshold_rule = "above_maximum_nonrisk_entropy"
    else:
        # Values at or above the cutoff would be counted.  Move one representable
        # value upward so ties at the cutoff cannot exceed the finite-sample cap.
        cutoff = nonrisk_entropy[len(nonrisk_entropy) - capacity - 1]
        entropy_threshold = float(np.nextafter(cutoff, np.inf))
        threshold_rule = "above_nonrisk_order_statistic"
    joint_triggered = [
        risk or entropy >= entropy_threshold
        for entropy, risk in zip(entropy_values, risk_triggered, strict=True)
    ]
    joint_count = sum(joint_triggered)
    if joint_count > allowed:
        raise RuntimeError(
            "deterministic entropy lock violated its own budget: "
            f"joint_episodes={joint_count}, allowed={allowed}"
        )
    return {
        "method": "clean_episode_joint_risk_entropy_order_statistic",
        "max_union_rate": float(max_union_rate),
        "episode_count": n,
        "allowed_joint_trigger_episodes": allowed,
        "risk_threshold": float(risk_threshold),
        "risk_trigger_episodes": int(risk_count),
        "remaining_entropy_capacity": int(capacity),
        "entropy_threshold": entropy_threshold,
        "entropy_threshold_rule": threshold_rule,
        "joint_trigger_episodes": int(joint_count),
        "joint_trigger_rate": float(joint_count / n),
        "risk_triggered_episode_ids": [
            str(row.get("episode_id", index))
            for index, (row, triggered) in enumerate(
                zip(clean_episode_summaries, risk_triggered, strict=True)
            )
            if triggered
        ],
    }


def trigger_from_prediction(
    prediction: Mapping[str, Any], *, risk_threshold: float, entropy_threshold: float
) -> dict[str, Any]:
    """Evaluate the locked joint trigger for one monitor prediction."""
    posterior = prediction.get("posterior")
    if not isinstance(posterior, Mapping):
        raise ValueError("monitor prediction must contain a posterior mapping")
    computed_entropy = normalized_posterior_entropy(posterior)
    if "normalized_entropy" in prediction and not np.isclose(
        float(prediction["normalized_entropy"]), computed_entropy, rtol=0.0, atol=1e-12
    ):
        raise ValueError("prediction normalized_entropy does not match its posterior")
    entropy = computed_entropy
    risk = float(prediction["risk"])
    posterior_risk = 1.0 - float(posterior["none"])
    # Predictions are emitted from float32 logits; permit only the resulting
    # serialization round-off, while still rejecting a materially inconsistent
    # risk field.
    if not np.isclose(risk, posterior_risk, rtol=0.0, atol=1e-6):
        raise ValueError("prediction risk does not match its posterior")
    if not (math.isfinite(risk) and math.isfinite(entropy)):
        raise ValueError("monitor prediction risk and entropy must be finite")
    if not 0.0 <= risk <= 1.0:
        raise ValueError(f"monitor prediction risk must be in [0, 1], got {risk}")
    if not math.isfinite(float(risk_threshold)) or not math.isfinite(float(entropy_threshold)):
        raise ValueError("trigger thresholds must be finite")
    risk_alarm = risk >= risk_threshold
    entropy_alarm = entropy >= entropy_threshold
    return {
        "risk": risk,
        "normalized_entropy": entropy,
        "risk_alarm": bool(risk_alarm),
        "entropy_alarm": bool(entropy_alarm),
        "joint_trigger": bool(risk_alarm or entropy_alarm),
    }


def mechanism_log_loss(posterior: Mapping[str, float], target: str) -> float:
    """Compute log loss for an offline mechanism label."""
    if target not in MECHANISMS:
        raise ValueError(f"unknown mechanism target: {target}")
    if set(posterior) != set(MECHANISMS):
        raise ValueError(
            "posterior mapping must contain exactly the monitor mechanisms: "
            f"{MECHANISMS}"
        )
    try:
        probabilities = np.asarray([posterior[name] for name in MECHANISMS], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("posterior mapping values must be numeric") from error
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("posterior probabilities must be finite and non-negative")
    if not np.isclose(float(probabilities.sum()), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("posterior probabilities must sum to one")
    probability = float(posterior[target])
    if probability <= 0.0:
        return float(-math.log(np.finfo(np.float64).tiny))
    return float(-math.log(probability))


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "paired_improvement_delta",
    seed_key: str = "seed",
    replicates: int = 10_000,
    seed: int = 1404,
) -> dict[str, Any]:
    """Bootstrap a mean while resampling independent scene-seed clusters."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    clusters: dict[str, list[float]] = {}
    for row in rows:
        cluster = str(row[seed_key])
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        clusters.setdefault(cluster, []).append(value)
    if not clusters:
        return {
            "independent_clusters": 0,
            "replicates": replicates,
            "mean": None,
            "interval_95": [None, None],
        }
    cluster_values = np.asarray(
        [float(np.mean(values)) for _, values in sorted(clusters.items())], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(cluster_values), size=len(cluster_values))
        draws[index] = float(cluster_values[sample].mean())
    return {
        "independent_clusters": int(len(cluster_values)),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(cluster_values.mean()),
        "interval_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }
