#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output = result.stdout.strip() or result.stderr.strip()
    return output if output else "unavailable"


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="SETUP_REPORT.md")
    parser.add_argument("--robosuite-repo", default="/root/autodl-tmp/VLA/src/robosuite")
    parser.add_argument("--robocasa-repo", default="/root/autodl-tmp/VLA/src/robocasa")
    return parser.parse_args()


def git_commit(path: str) -> str:
    if not (Path(path) / ".git").exists():
        return "unavailable"
    return command_output(["git", "-C", path, "rev-parse", "HEAD"])


def main() -> int:
    args = parse_args()
    disk = shutil.disk_usage("/root/autodl-tmp")
    lines = [
        "# AutoDL setup report",
        "",
        "## Hardware and storage",
        "",
        "```text",
        command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        command_output(["bash", "-lc", "free -h"]),
        f"/root/autodl-tmp free: {disk.free / (1024**3):.1f} GiB",
        "```",
        "",
        "## Software",
        "",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Platform: `{platform.platform()}`",
        f"- MuJoCo: `{package_version('mujoco')}`",
        f"- Gymnasium: `{package_version('gymnasium')}`",
        f"- robosuite: `{package_version('robosuite')}`",
        f"- RoboCasa: `{package_version('robocasa')}`",
        f"- Benchmark package: `{package_version('vla-recovery-bench')}`",
        f"- Conda prefix: `{sys.prefix}`",
        f"- robosuite commit: `{git_commit(args.robosuite_repo)}`",
        f"- RoboCasa commit: `{git_commit(args.robocasa_repo)}`",
        "",
        "## Smoke test",
        "",
        "Run the following twice and attach each generated `smoke_test.json`:",
        "",
        "```bash",
        "python scripts/smoke_test_robocasa.py --output $VLA_ROOT/outputs/smoke_test_run1",
        "python scripts/smoke_test_robocasa.py --output $VLA_ROOT/outputs/smoke_test_run2",
        "```",
        "",
        "## Status",
        "",
        "RoboCasa installation and the audited frozen-policy integration are complete.",
        "Scientific fault evaluation remains blocked until the v1.4 firewall, artifact,",
        "identifiability-pilot, and randomized-branch gates pass.",
        "",
    ]
    output = Path(args.output)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
