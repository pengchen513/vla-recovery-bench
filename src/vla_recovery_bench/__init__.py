"""Failure injection and recovery evaluation for frozen VLA policies."""

from .metrics import aggregate_episode_metrics
from .runner import ExperimentRunner, RunnerConfig

__all__ = ["ExperimentRunner", "RunnerConfig", "aggregate_episode_metrics"]
__version__ = "0.1.0"

