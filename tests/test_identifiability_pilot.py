import unittest

import numpy as np

from scripts.analyze_identifiability_pilot import (
    _binary_metrics,
    _cluster_bootstrap,
    _wilson_interval,
)
from scripts.run_identifiability_pilot import _image_features, _monitor_features, _summarize
from vla_recovery_bench.robocasa_adapter import RoboCasaEnvironment
from vla_recovery_bench.types import ActionChunkMetadata, MonitorContext


class IdentifiabilityPilotTest(unittest.TestCase):
    def test_image_features_are_shape_and_range_stable(self) -> None:
        previous = np.full((8, 8, 3), 100, dtype=np.uint8)
        current = previous.copy()
        current[2:6, 2:6] = 0
        features = _image_features(current, previous)
        self.assertGreater(features["zero_fraction"], 0.0)
        self.assertGreater(features["temporal_abs_difference"], 0.0)

    def test_monitor_features_use_only_numeric_observation_and_action(self) -> None:
        previous = {
            "video": {"cam": np.full((4, 4, 3), 100, dtype=np.uint8)},
            "state": {"x": np.zeros(2, dtype=np.float32)},
        }
        current = {
            "video": {"cam": np.zeros((4, 4, 3), dtype=np.uint8)},
            "state": {"x": np.ones(2, dtype=np.float32)},
        }
        context = MonitorContext(
            episode_id=0,
            step=0,
            instruction="task",
            previous_observation=previous,
            observation=current,
            action={"a": np.ones(1, dtype=np.float32)},
            action_chunk=(),
            chunk=ActionChunkMetadata(),
        )
        features = _monitor_features(context)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        self.assertIn("observation_evidence", features)
        self.assertIn("actuator_evidence", features)

    def test_after_step_hook_consumes_exactly_one_corrupted_observation(self) -> None:
        environment = RoboCasaEnvironment.__new__(RoboCasaEnvironment)
        environment._occlusion_steps = 1
        environment._occlusion_variant = "all_zero"
        environment._occlusion_keys = {"video.cam"}
        environment._last_image_frames = {"video.cam": np.full((4, 4, 3), 100, dtype=np.uint8)}
        environment._pre_step_image_frames = dict(environment._last_image_frames)
        current = {"video": {"cam": np.full((4, 4, 3), 200, dtype=np.uint8)}}
        transformed = environment.apply_pending_observation_fault(current)
        np.testing.assert_array_equal(transformed["video"]["cam"], 0)
        self.assertEqual(environment._occlusion_steps, 0)

    def test_summary_uses_only_declared_exposure_window(self) -> None:
        def row(condition: str, evidence: list[float]) -> dict[str, object]:
            return {
                "condition": condition,
                "success": False,
                "steps": 10,
                "reward": 0.0,
                "configured_fault_count": 1,
                "applied_fault_count": 1,
                "exposed_fault_count": 1,
                "not_exposed": False,
                "alarm_count": 0,
                "detection_delay_steps": None,
                "first_affected_input_step": 2,
                "exposure_end_step_exclusive": 4,
                "analysis_steps": [
                    {
                        "control_step": index,
                        "observation_step": index,
                        "observation_evidence": value,
                    }
                    for index, value in enumerate(evidence)
                ],
            }

        actuator = row("actuator_fault", [0.0, 0.0, 0.1, 0.1, 1.0])
        observation = row("observation_fault", [0.0, 0.0, 0.9, 0.9, 0.0])
        result = _summarize([actuator, observation])
        self.assertEqual(
            result["exploratory_passive_rule"]["balanced_accuracy"],
            1.0,
        )

    def test_summary_does_not_count_alarm_after_exposure(self) -> None:
        row = {
            "condition": "actuator_fault",
            "success": False,
            "steps": 10,
            "reward": 0.0,
            "configured_fault_count": 1,
            "applied_fault_count": 1,
            "exposed_fault_count": 1,
            "not_exposed": False,
            "alarm_count": 1,
            "detection_delay_steps": None,
            "first_affected_input_step": 2,
            "exposure_end_step_exclusive": 4,
            "analysis_steps": [
                {
                    "control_step": index,
                    "observation_step": index,
                    "observation_evidence": 0.0,
                }
                for index in range(5)
            ],
        }
        result = _summarize([row])
        self.assertEqual(
            result["conditions"]["actuator_fault"]["exposure_window_detections"],
            0,
        )

    def test_offline_analysis_metrics_and_cluster_bootstrap(self) -> None:
        rows = [
            {"pair_id": "a", "actual": "actuator_fault", "predicted": "actuator_fault"},
            {
                "pair_id": "a",
                "actual": "observation_fault",
                "predicted": "observation_fault",
            },
            {"pair_id": "b", "actual": "actuator_fault", "predicted": "actuator_fault"},
            {
                "pair_id": "b",
                "actual": "observation_fault",
                "predicted": "observation_fault",
            },
        ]
        metrics = _binary_metrics(rows)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        interval = _cluster_bootstrap(rows, replicates=100, seed=7)
        self.assertEqual(interval["independent_clusters"], 2)
        self.assertEqual(interval["balanced_accuracy"], [1.0, 1.0])

    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = _wilson_interval(11, 12)
        self.assertLess(lower, 11 / 12)
        self.assertGreater(upper, 11 / 12)


if __name__ == "__main__":
    unittest.main()
