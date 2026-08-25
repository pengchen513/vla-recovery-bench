from __future__ import annotations

from collections.abc import Iterable
from statistics import fmean

from .types import EpisodeResult


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def aggregate_episode_metrics(results: Iterable[EpisodeResult]) -> dict[str, float | int]:
    episodes = list(results)
    total = len(episodes)
    successes = sum(result.success for result in episodes)
    fault_count = sum(result.fault_count for result in episodes)
    detected = sum(result.detected_fault_count for result in episodes)
    false_alarms = sum(result.false_alarm_count for result in episodes)
    missed = max(fault_count - detected, 0)
    precision = _divide(detected, detected + false_alarms)
    recall = _divide(detected, detected + missed)
    f1 = _divide(2 * precision * recall, precision + recall)
    recovery_attempts = sum(result.recovery_attempted for result in episodes)
    recovery_successes = sum(result.recovery_success for result in episodes)
    delays = [delay for result in episodes for delay in result.detection_delays]

    return {
        "episodes": total,
        "successful_episodes": successes,
        "success_rate": _divide(successes, total),
        "mean_episode_steps": fmean(result.steps for result in episodes) if episodes else 0.0,
        "fault_count": fault_count,
        "detected_fault_count": detected,
        "missed_fault_count": missed,
        "false_alarm_count": false_alarms,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "mean_detection_delay_steps": fmean(delays) if delays else 0.0,
        "recovery_attempted_episodes": recovery_attempts,
        "recovery_successful_episodes": recovery_successes,
        "recovery_success_rate": _divide(recovery_successes, recovery_attempts),
    }
