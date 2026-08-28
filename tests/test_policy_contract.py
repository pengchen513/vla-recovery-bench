import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.check_policy_contract import _checkpoint_set_sha256, check_manifest


class PolicyContractTest(unittest.TestCase):
    @staticmethod
    def _complete_manifest(directory: str) -> tuple[dict, dict]:
        checkpoint = Path(directory) / "model.bin"
        checkpoint.write_bytes(b"frozen-policy")
        sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        action_space = {
            "type": "gymnasium.spaces.dict.Dict",
            "shape": None,
            "dtype": None,
            "spaces": {
                "action.delta": {
                    "type": "gymnasium.spaces.box.Box",
                    "shape": [2],
                    "dtype": "float32",
                    "low": [-1.0, -1.0],
                    "high": [1.0, 1.0],
                }
            },
        }
        record = {"path": checkpoint.name, "size": checkpoint.stat().st_size, "sha256": sha256}
        manifest = {
            "embodiment": "PandaOmron",
            "camera_keys": ["video.cam"],
            "image_shape": [256, 256, 3],
            "image_preprocessing": {"source_shape": [256, 256, 3], "resize": [224, 224]},
            "proprioception_keys": ["state.pose"],
            "state_normalization": {"continuous": "min_max"},
            "action_space": action_space,
            "action_keys": ["action.delta"],
            "action_dim": 2,
            "model_action_dim": 4,
            "action_horizon": 16,
            "control_mode": "OSC_POSE",
            "action_normalization": {"continuous": "min_max"},
            "prompt_key": "annotation.human.task_description",
            "checkpoint_path": directory,
            "checkpoint_files": [record],
            "checkpoint_sha256": _checkpoint_set_sha256([record]),
            "published_clean_baseline": {"episodes": 30},
        }
        probe = {
            "observation": {
                "contract": [
                    {"key": "video.cam", "shape": [256, 256, 3]},
                    {"key": "state.pose", "shape": [3]},
                    {"key": "annotation.human.task_description", "shape": []},
                ],
                "images": [{"key": "video.cam"}],
            },
            "action": {"space": action_space},
        }
        return manifest, probe

    def test_missing_fields_block(self) -> None:
        errors = check_manifest({}, {})
        self.assertTrue(any("missing required field" in error for error in errors))

    def test_checkpoint_set_hash_is_order_independent(self) -> None:
        records = [
            {"path": "b", "sha256": "2"},
            {"path": "a", "sha256": "1"},
        ]
        self.assertEqual(
            _checkpoint_set_sha256(records),
            _checkpoint_set_sha256(list(reversed(records))),
        )

    def test_missing_checkpoint_shard_blocks(self) -> None:
        with TemporaryDirectory() as directory:
            manifest, probe = self._complete_manifest(directory)
            manifest["checkpoint_files"] = [{"path": "missing", "size": 1, "sha256": "0"}]
            manifest["checkpoint_sha256"] = _checkpoint_set_sha256(manifest["checkpoint_files"])
            errors = check_manifest(manifest, probe)
        expected = str(Path(directory) / "missing")
        self.assertTrue(any(expected in error for error in errors))

    def test_complete_manifest_and_probe_pass(self) -> None:
        with TemporaryDirectory() as directory:
            manifest, probe = self._complete_manifest(directory)
            self.assertEqual(check_manifest(manifest, probe), [])

    def test_missing_preprocessing_blocks(self) -> None:
        with TemporaryDirectory() as directory:
            manifest, probe = self._complete_manifest(directory)
            del manifest["image_preprocessing"]
            errors = check_manifest(manifest, probe)
        self.assertTrue(any("image_preprocessing" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
