import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from vla_recovery_bench.monitor import FEATURE_NAMES
from vla_recovery_bench.monitor_dataset import (
    MonitorDatasetWriter,
    load_monitor_dataset,
)


class MonitorDatasetTest(unittest.TestCase):
    def test_input_and_privileged_label_channels_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with MonitorDatasetWriter(
                output, partition="train", protocol_sha256="abc"
            ) as writer:
                writer.write_episode(
                    token="opaque-token",
                    features=np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32),
                    control_steps=np.arange(3),
                    observation_steps=np.arange(1, 4),
                    instruction="task",
                    label={
                        "mechanism": "actuator_fault",
                        "condition": "actuator_fault",
                        "seed": 600,
                        "pair_id": "scene-600",
                        "success": False,
                        "reward": 0.0,
                        "fault_schedule": [{"kind": "actuator_variant"}],
                    },
                    exposure=np.asarray([False, True, True]),
                )
            with h5py.File(output / "monitor_inputs.h5", "r") as stream:
                group = stream["episodes"]["opaque-token"]
                self.assertNotIn("mechanism", group.attrs)
                self.assertNotIn("seed", group.attrs)
                self.assertNotIn("reward", group.attrs)
                self.assertNotIn("exposure", group)
            episodes = load_monitor_dataset(output, expected_partition="train")
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].mechanism, "actuator_fault")
            np.testing.assert_array_equal(episodes[0].exposure, [False, True, True])

    def test_existing_dataset_channel_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "monitor_inputs.h5").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                MonitorDatasetWriter(output, partition="train", protocol_sha256="abc")

    def test_reader_rejects_scalar_feature_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with MonitorDatasetWriter(
                output, partition="train", protocol_sha256="abc"
            ) as writer:
                writer.write_episode(
                    token="opaque-token",
                    features=np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
                    control_steps=np.asarray([0]),
                    observation_steps=np.asarray([1]),
                    instruction="task",
                    label={
                        "mechanism": "none",
                        "condition": "clean",
                        "seed": 600,
                        "pair_id": "scene-600",
                        "success": False,
                    },
                    exposure=np.asarray([False]),
                )
            with h5py.File(output / "monitor_inputs.h5", "r+") as stream:
                group = stream["episodes"]["opaque-token"]
                del group["features"]
                group.create_dataset("features", data=np.asarray(0.0, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "features must be a matrix"):
                load_monitor_dataset(output, expected_partition="train")


if __name__ == "__main__":
    unittest.main()
