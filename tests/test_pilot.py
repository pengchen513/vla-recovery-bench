import json
import tempfile
import unittest
from pathlib import Path

from vla_recovery_bench.pilot import pilot_episode_plan, validate_pilot_artifacts


class PilotPlanTest(unittest.TestCase):
    def test_v14_expands_crossed_rows(self) -> None:
        config = json.loads(
            (Path(__file__).parents[1] / "configs/identifiability_pilot_v1_4.json").read_text()
        )
        plan = pilot_episode_plan(config)
        self.assertEqual(len(plan), 36)
        for seed in config["seed_block"]["seeds"]:
            rows = [item for item in plan if item["seed"] == seed]
            self.assertEqual(
                {item["condition"] for item in rows},
                {"clean", "actuator_fault", "observation_fault"},
            )
            observation = next(item for item in rows if item["condition"] == "observation_fault")
            factor = next(item for item in config["scene_seed_factors"] if item["seed"] == seed)
            fault = observation["faults"][0]
            self.assertEqual(fault.step, factor["onset_step"] - 1)
            self.assertEqual(fault.phase.value, "after_step")

    def test_artifact_validator_detects_online_label_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            required_json = {
                "run_manifest.json": {
                    "protocol_version": "1.4",
                    "environment": {},
                    "policy": {},
                    "monitor": {},
                    "seeds": [1],
                    "episode_plan": [{}, {}, {}],
                },
                "metrics.json": {"status": "completed", "episode_count": 3},
                "monitor_config.json": {},
                "calibration.json": {},
                "software_versions.json": {},
                "policy_state_before.json": {
                    "current_parameter_sha256": "same",
                    "model_training": False,
                    "all_parameters_frozen": True,
                },
                "policy_state_after.json": {
                    "current_parameter_sha256": "same",
                    "model_training": False,
                    "all_parameters_frozen": True,
                },
            }
            for name, value in required_json.items():
                (output / name).write_text(json.dumps(value), encoding="utf-8")
            episodes = [
                {"episode_id": condition, "condition": condition}
                for condition in ("clean", "actuator_fault", "observation_fault")
            ]
            (output / "episodes.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in episodes),
                encoding="utf-8",
            )
            (output / "monitor_stream.jsonl").write_text(
                json.dumps(
                    {
                        "episode_token": "x",
                        "condition": "clean",
                        "chunk": {"chunk_length": 1},
                        "requested_action_chunk": [{}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output / "audit_stream.jsonl").write_text("{}\n", encoding="utf-8")
            errors = validate_pilot_artifacts(output, expected_episode_count=3)
            self.assertTrue(any("leaks fields" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
