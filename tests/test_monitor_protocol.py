import json
import unittest
from pathlib import Path

from vla_recovery_bench.monitor_protocol import (
    CONDITIONS,
    factor_balance,
    monitor_episode_plan,
    validate_monitor_protocol,
    validate_probe_protocol,
)

ROOT = Path(__file__).parents[1]


class MonitorProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (ROOT / "configs/monitor_training_v1_0.json").read_text()
        )
        self.probe = json.loads(
            (ROOT / "configs/diagnostic_probe_v1_0.json").read_text()
        )

    def test_checked_in_monitor_and_probe_protocols_pass(self) -> None:
        self.assertEqual(validate_monitor_protocol(self.config), [])
        self.assertEqual(validate_probe_protocol(self.probe, self.config), [])

    def test_seed_splits_are_disjoint_and_final_test_is_not_collectable(self) -> None:
        self.config["splits"]["final_test_scene_seeds"][0] = self.config["splits"][
            "train_scene_seeds"
        ][0]
        errors = validate_monitor_protocol(self.config)
        self.assertTrue(any("appears in both" in error for error in errors))
        with self.assertRaisesRegex(ValueError, "unsupported monitor partition"):
            monitor_episode_plan(self.config, "pilot")

    def test_training_plan_crosses_mechanisms_without_factor_confounding(self) -> None:
        plan = monitor_episode_plan(self.config, "train", seeds=[600, 601])
        self.assertEqual(len(plan), 6)
        for seed in (600, 601):
            rows = [item for item in plan if item["seed"] == seed]
            self.assertEqual({item["condition"] for item in rows}, set(CONDITIONS))
            actuator = next(item for item in rows if item["condition"] == "actuator_fault")
            observation = next(
                item for item in rows if item["condition"] == "observation_fault"
            )
            self.assertEqual(actuator["factor_row"], observation["factor_row"])
            onset = actuator["factor_row"]["onset_step"]
            self.assertEqual(actuator["faults"][0].step, onset)
            self.assertEqual(observation["faults"][0].step, onset - 1)

    def test_factor_assignment_is_deterministic_and_marginally_balanced(self) -> None:
        first = monitor_episode_plan(self.config, "calibration")
        second = monitor_episode_plan(self.config, "calibration")
        self.assertEqual(first, second)
        balance = factor_balance(self.config, "train")
        self.assertEqual(
            set(balance["actuator_variant"]),
            set(self.config["fault_sampling"]["actuator_variants"]),
        )
        counts = list(balance["onset_step"].values())
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_probe_cannot_drop_calibration_gate(self) -> None:
        self.probe["gates"]["probe_must_not_run_before_monitor_calibration"] = False
        errors = validate_probe_protocol(self.probe, self.config)
        self.assertIn("probe must remain blocked until monitor calibration", errors)


if __name__ == "__main__":
    unittest.main()
