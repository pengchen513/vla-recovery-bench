#!/usr/bin/env python3
"""Run the deterministic, metadata-only v1.4 power simulation.

The simulation models the single primary endpoint (paired binary recovery) and
does not import RoboCasa or load a policy. It is intentionally conservative:
invalid runs and attrition reduce the effective paired-unit count, and power is
the fraction of replicates passing the exact two-sided McNemar test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.stats import binomtest as _scipy_binomtest
except ImportError:  # pragma: no cover - exercised on minimal environments
    _scipy_binomtest = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/power_analysis_v1_4.json"


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _joint_probabilities(
    p_control: float, p_treatment: float, correlation: float
) -> tuple[float, float, float, float]:
    """Return (both, control-only, treatment-only, neither) probabilities."""
    covariance = correlation * math.sqrt(
        p_control * (1 - p_control) * p_treatment * (1 - p_treatment)
    )
    both = p_control * p_treatment + covariance
    lower = max(0.0, p_control + p_treatment - 1.0)
    upper = min(p_control, p_treatment)
    both = min(max(both, lower), upper)
    return (
        both,
        p_control - both,
        p_treatment - both,
        1.0 - p_control - p_treatment + both,
    )


def _mcnemar_p_value(treatment_only: int, control_only: int) -> float:
    discordant = treatment_only + control_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(treatment_only, control_only) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _mcnemar_p_value_scipy(treatment_only: int, control_only: int) -> float:
    """Use the canonical exact implementation when SciPy is available."""
    if _scipy_binomtest is None:
        return _mcnemar_p_value(treatment_only, control_only)
    discordant = treatment_only + control_only
    if discordant == 0:
        return 1.0
    return float(_scipy_binomtest(treatment_only, discordant, 0.5).pvalue)


def _simulate_scenario(
    *,
    units: int,
    baseline: float,
    correlation: float,
    effect: float,
    invalid_rate: float,
    attrition_rate: float,
    replicates: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    treatment = min(1.0, baseline + effect)
    probabilities = _joint_probabilities(baseline, treatment, correlation)
    passed = 0
    for _ in range(replicates):
        valid = rng.random(units) >= invalid_rate
        valid &= rng.random(units) >= attrition_rate
        draws = rng.choice(4, size=units, p=probabilities)
        draws = draws[valid]
        treatment_only = int(np.count_nonzero(draws == 2))
        control_only = int(np.count_nonzero(draws == 1))
        if _mcnemar_p_value_scipy(treatment_only, control_only) < alpha:
            passed += 1
    return passed / replicates


def run_power_analysis(config: dict[str, Any]) -> dict[str, Any]:
    sample_rule = config["sample_size_rule"]
    scenarios = sample_rule["sensitivity_scenarios"]
    replicates = int(sample_rule["replicates"])
    seed = int(sample_rule["seed"])
    alpha = float(config["alpha_two_sided"])
    invalid_rate = float(config["invalid_run_rate"])
    attrition_rate = float(config["attrition_allowance"])
    target_power = float(config["target_power"])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for units in sample_rule["candidate_independent_units"]:
        powers = []
        for scenario in scenarios:
            power = _simulate_scenario(
                units=int(units),
                baseline=float(scenario["baseline_recovery_rate"]),
                correlation=float(scenario["within_pair_correlation"]),
                effect=float(scenario["effect"]),
                invalid_rate=invalid_rate,
                attrition_rate=attrition_rate,
                replicates=replicates,
                alpha=alpha,
                rng=rng,
            )
            powers.append(power)
        rows.append(
            {
                "independent_units": int(units),
                "scenario_power": powers,
                "minimum_power": min(powers),
                "passes_all_scenarios": min(powers) >= target_power,
            }
        )
    selected = next(
        (row["independent_units"] for row in rows if row["passes_all_scenarios"]),
        None,
    )
    return {
        "status": "completed",
        "scientific_result": False,
        "protocol_version": config["protocol_version"],
        "primary_endpoint": config["primary_endpoint"],
        "simulation": {
            "method": sample_rule["method"],
            "seed": seed,
            "replicates": replicates,
            "alpha_two_sided": alpha,
            "invalid_run_rate": invalid_rate,
            "attrition_allowance": attrition_rate,
            "test": "exact_two_sided_mcnemar",
        },
        "candidate_results": rows,
        "selected_independent_units": selected,
        "selection_rule": sample_rule["select"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/pc/VLA/outputs/power_analysis_v1_4.json"),
    )
    args = parser.parse_args()
    # Fail before the long Monte Carlo run when an immutable artifact exists.
    if args.output.exists() and args.output.stat().st_size > 0:
        raise FileExistsError(f"refusing to overwrite non-empty artifact: {args.output}")
    result = run_power_analysis(_load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
