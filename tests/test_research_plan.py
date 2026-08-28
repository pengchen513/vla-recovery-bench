import unittest

from scripts.validate_research_plan import (
    validate_all,
    validate_intervention,
    validate_pilot,
)


class ResearchPlanTest(unittest.TestCase):
    def test_checked_in_plan_is_consistent(self) -> None:
        self.assertEqual(validate_all(), [])

    def test_pilot_rejects_clean_in_primary_diagnosis(self) -> None:
        pilot = {
            "protocol_version": "1.4",
            "seed_block": {"seeds": list(range(12)), "episodes_per_seed": 3},
            "episode_count": 36,
            "scene_seed_factors": [
                {
                    "seed": seed,
                    "actuator_variant": "arm_only",
                    "observation_variant": "partial_mask",
                    "observation_camera": "video.robot0_agentview_left",
                    "onset_step": 80,
                    "duration_steps": 4,
                }
                for seed in range(12)
            ],
            "balance_requirements": {
                "each_scene_seed_reused_across_fault_mechanisms": True
            },
            "analysis": {
                "clean_use": "wrong",
                "all_zero_conditions_are_primary_evidence": False,
                "pilot_is_recovery_evidence": False,
            },
        }
        self.assertIn(
            "clean episodes may only support false-intervention and calibration estimates",
            validate_pilot(pilot),
        )

    def test_intervention_plan_rejects_event_only_budget(self) -> None:
        protocol = {
            "protocol_version": "1.3",
            "assignment": {
                "randomization_seed": "sha256(pair_id || decision_step || protocol_version)",
                "primary_arms": {"selector": 0.5, "fixed_retry": 0.5},
            },
            "cost_vector": {
                "elapsed_execution_time_ms": {},
                "extra_compute_ms": {},
                "peak_memory_mb": {},
                "human_help_count": {},
                "risk_penalty": {},
            },
            "analysis": {"clean_budget_is_event_count": True},
            "implementation_gate": {
                "state_clone_test": "required_for_causal_common_prefix_claim"
            },
        }
        self.assertIn(
            "clean budget cannot be represented only as event count",
            validate_intervention(protocol),
        )

    def test_v14_pilot_rejects_mechanism_confounded_assignment(self) -> None:
        import json
        from pathlib import Path

        pilot = json.loads(
            (Path(__file__).parents[1] / "configs/identifiability_pilot_v1_4.json").read_text()
        )
        pilot["scene_seed_factors"][1]["onset_step"] = 240
        errors = validate_pilot(pilot)
        self.assertTrue(any("onset_step" in error for error in errors))

    def test_v14_pilot_rejects_observation_timing_drift(self) -> None:
        import json
        from pathlib import Path

        pilot = json.loads(
            (Path(__file__).parents[1] / "configs/identifiability_pilot_v1_4.json").read_text()
        )
        pilot["timing_contract"]["observation_fault_phase"] = "before_action"
        self.assertIn("observation fault phase must be after_step", validate_pilot(pilot))


if __name__ == "__main__":
    unittest.main()
