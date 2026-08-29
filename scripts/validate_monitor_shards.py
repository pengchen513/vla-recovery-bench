#!/usr/bin/env python3
"""Read-only formal monitor-shard integrity gate.

This command never loads a VLA checkpoint, starts RoboCasa, trains a model, or
modifies a shard. It returns 0 only when the supplied shard set exactly covers
the frozen protocol partition and every checksum and provenance check passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_recovery_bench.monitor_gate import (
    FORMAL_PARTITIONS,
    format_gate_failure,
    validate_formal_shard_set,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/monitor_relock_v1_2.json"
DEFAULT_MANIFEST = ROOT / "configs/policies/groot_n1_5_robocasa_atomic_seen_30p.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--partition", choices=FORMAL_PARTITIONS, required=True)
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--json", action="store_true", help="print the complete JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_formal_shard_set(
        args.protocol,
        args.manifest,
        partition=args.partition,
        shard_paths=args.shard,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["passed"]:
        print(
            "formal shard integrity gate passed: "
            f"partition={args.partition}, shards={report['shard_count']}, "
            f"seeds={report['observed_seed_count']}, "
            f"episodes={report['observed_episode_count']}"
        )
    else:
        print(format_gate_failure(report))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
