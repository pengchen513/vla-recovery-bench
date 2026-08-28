import tempfile
import unittest
from pathlib import Path

import numpy as np

from vla_recovery_bench.groot_adapter import (
    ACTION_DIMS,
    ACTION_HORIZON,
    CAMERA_SHAPES,
    PROMPT_KEY,
    STATE_SHAPES,
)
from vla_recovery_bench.monitor import (
    FEATURE_NAMES,
    MECHANISMS,
    FaultConditionedTemporalMonitor,
    context_to_feature,
)
from vla_recovery_bench.types import ActionChunkMetadata, MonitorContext


def observation(value: int = 100) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        key: np.full(shape, value, dtype=np.uint8) for key, shape in CAMERA_SHAPES.items()
    }
    result.update(
        {key: np.zeros(shape, dtype=np.float32) for key, shape in STATE_SHAPES.items()}
    )
    result[PROMPT_KEY] = np.asarray(["pick and place"])
    return result


def action(index: int) -> dict[str, np.ndarray]:
    return {
        key: np.full(width, index / 100.0, dtype=np.float32)
        for key, width in ACTION_DIMS.items()
    }


class FaultMonitorTest(unittest.TestCase):
    def test_context_feature_preserves_exact_contract_and_chunk_position(self) -> None:
        chunk = tuple(action(index) for index in range(ACTION_HORIZON))
        context = MonitorContext(
            episode_id=0,
            step=4,
            instruction="pick and place",
            previous_observation=observation(100),
            observation=observation(110),
            action=chunk[3],
            action_chunk=chunk,
            chunk=ActionChunkMetadata(
                chunk_id=0,
                position_in_chunk=3,
                chunk_length=ACTION_HORIZON,
                remaining_horizon=100,
            ),
        )
        feature = context_to_feature(context)
        self.assertEqual(feature.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_missing_observation_or_incomplete_chunk_fails_closed(self) -> None:
        current = observation()
        current.pop(next(iter(CAMERA_SHAPES)))
        chunk = tuple(action(index) for index in range(ACTION_HORIZON))
        context = MonitorContext(
            episode_id=0,
            step=0,
            instruction="pick and place",
            previous_observation=observation(),
            observation=current,
            action=chunk[0],
            action_chunk=chunk,
            chunk=ActionChunkMetadata(chunk_length=ACTION_HORIZON),
        )
        with self.assertRaisesRegex(ValueError, "observation keys"):
            context_to_feature(context)
        good = MonitorContext(
            episode_id=0,
            step=0,
            instruction="pick and place",
            previous_observation=observation(),
            observation=observation(),
            action=chunk[0],
            action_chunk=chunk[:-1],
            chunk=ActionChunkMetadata(chunk_length=ACTION_HORIZON),
        )
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            context_to_feature(good)

    def test_train_calibrate_save_and_reload(self) -> None:
        rng = np.random.default_rng(8)
        row_count = 18
        features = []
        labels = []
        episode_ids = []
        for label, offset in enumerate((-0.7, 0.0, 0.7)):
            rows = rng.normal(offset, 0.1, size=(row_count, len(FEATURE_NAMES))).astype(
                np.float32
            )
            features.append(rows)
            labels.extend([label] * row_count)
            episode_ids.extend([f"episode-{label}"] * row_count)
        monitor = FaultConditionedTemporalMonitor(window_size=4, seed=11)
        report = monitor.fit(
            np.concatenate(features),
            np.asarray(labels),
            episode_ids=episode_ids,
            epochs=12,
        )
        self.assertEqual(set(report["class_counts"]), set(MECHANISMS))
        calibration = monitor.calibrate_clean_episode_maxima([features[0][:8]])
        self.assertEqual(calibration["status"], "calibrated")
        monitor.reset()
        prediction = monitor.predict_features(features[1][0])
        self.assertAlmostEqual(sum(prediction["posterior"].values()), 1.0, places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "monitor.npz"
            monitor.save(path)
            loaded = FaultConditionedTemporalMonitor.load(path)
            loaded.reset()
            restored = loaded.predict_features(features[1][0])
            self.assertAlmostEqual(prediction["risk"], restored["risk"], places=6)


if __name__ == "__main__":
    unittest.main()
