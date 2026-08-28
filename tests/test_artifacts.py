import json
import tempfile
import unittest
from pathlib import Path

from vla_recovery_bench.artifacts import (
    ensure_empty_output_dir,
    validate_json_artifact_contract,
    write_json_once,
)


class ArtifactContractTest(unittest.TestCase):
    def test_output_directory_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            ensure_empty_output_dir(path)
            (path / "existing").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(path)

    def test_json_write_once_and_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json_once(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_json_once(path, {"ok": False})
            run = Path(directory) / "run"
            run.mkdir()
            for name in (
                "episodes.jsonl",
                "metrics.json",
                "monitor_config.json",
                "calibration.json",
                "software_versions.json",
                "policy_state_before.json",
                "policy_state_after.json",
            ):
                (run / name).write_text("{}\n", encoding="utf-8")
            manifest = {
                "protocol_version": "1.4",
                "environment": {},
                "policy": {},
                "monitor": {},
                "seeds": [],
            }
            (run / "run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            self.assertEqual(validate_json_artifact_contract(run), [])


if __name__ == "__main__":
    unittest.main()
