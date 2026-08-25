from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dummy import (
    ConstantFrozenPolicy,
    DummyProgressEnvironment,
    ProgressFailureMonitor,
    RetryRecoveryController,
    require_mapping,
)
from .faults import FaultSchedule, fault_specs_from_config
from .metrics import aggregate_episode_metrics
from .recording import JsonlRecorder
from .runner import ExperimentRunner, RunnerConfig


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("configuration root must be a JSON object")
    return value


def run_dummy(config_path: str, output: str) -> int:
    config = _load_config(config_path)
    monitor_config = require_mapping(config.get("monitor", {}), "monitor")
    recovery_config = require_mapping(config.get("recovery", {}), "recovery")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with JsonlRecorder(output_dir / "events.jsonl") as recorder:
        runner = ExperimentRunner(
            environment=DummyProgressEnvironment(int(config.get("goal_progress", 8))),
            policy=ConstantFrozenPolicy(),
            monitor=ProgressFailureMonitor(
                minimum_progress_delta=float(
                    monitor_config.get("minimum_progress_delta", 0.5)
                ),
                stagnation_patience=int(monitor_config.get("stagnation_patience", 1)),
            ),
            recovery=RetryRecoveryController(
                max_retries=int(recovery_config.get("max_retries", 4))
            ),
            fault_schedule=FaultSchedule(fault_specs_from_config(config.get("faults", []))),
            recorder=recorder,
            config=RunnerConfig(
                episodes=int(config.get("episodes", 20)),
                horizon=int(config.get("horizon", 14)),
                base_seed=int(config.get("seed", 0)),
                instruction="reach the target progress",
                detection_match_window=int(config.get("detection_match_window", 5)),
            ),
        )
        results = runner.run()
        metrics = aggregate_episode_metrics(results)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Events: {output_dir / 'events.jsonl'}")
    print(f"Metrics: {metrics_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-recovery-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dummy = subparsers.add_parser("dummy", help="run the deterministic local experiment")
    dummy.add_argument("--config", default="configs/local_dummy.json")
    dummy.add_argument("--output", default="outputs/local_dummy")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "dummy":
        return run_dummy(args.config, args.output)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
