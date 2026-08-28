import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vla_recovery_bench.monitor import FEATURE_NAMES, FEATURE_VERSION
from vla_recovery_bench.monitor_dataset import MonitorDatasetWriter
from vla_recovery_bench.monitor_gate import (
    CHUNK_LENGTH_INDEX,
    REMAINING_HORIZON_INDEX,
    build_shard_integrity_manifest,
    episode_token,
    sha256_file,
    validate_formal_shard_set,
)
from vla_recovery_bench.monitor_protocol import CONDITIONS, monitor_episode_plan
from vla_recovery_bench.recording import to_jsonable

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "configs/monitor_training_v1_0.json"
POLICY_PATH = ROOT / "configs/policies/groot_n1_5_robocasa_atomic_seen_30p.json"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(to_jsonable(value), sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _zero_action(action_space: dict[str, Any]) -> dict[str, list[float]]:
    return {
        key: [0.0] * int(np.prod(contract["shape"]))
        for key, contract in action_space["spaces"].items()
    }


class MonitorFormalShardGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.policy_manifest = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    def _write_formal_shard(
        self,
        directory: Path,
        *,
        partition: str,
        seeds: list[int],
    ) -> None:
        directory.mkdir()
        protocol_sha256 = sha256_file(PROTOCOL_PATH)
        policy_manifest_sha256 = sha256_file(POLICY_PATH)
        plan = monitor_episode_plan(self.protocol, partition, seeds=seeds)
        action_space = self.policy_manifest["action_space"]
        action = _zero_action(action_space)
        index_records: list[dict[str, Any]] = []
        episode_records: list[dict[str, Any]] = []
        audit_records: list[dict[str, Any]] = []
        with MonitorDatasetWriter(
            directory,
            partition=partition,
            protocol_sha256=protocol_sha256,
        ) as writer:
            for index, item in enumerate(plan):
                token = episode_token(protocol_sha256, item["episode_id"])
                is_clean = item["condition"] == "clean"
                features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
                features[0, CHUNK_LENGTH_INDEX] = 16
                features[0, REMAINING_HORIZON_INDEX] = (
                    self.protocol["environment"]["horizon"] - 1
                )
                writer.write_episode(
                    token=token,
                    features=features,
                    control_steps=np.asarray([0], dtype=np.int32),
                    observation_steps=np.asarray([1], dtype=np.int32),
                    instruction="Pick the object and place it in the cabinet.",
                    label={
                        "episode_id": item["episode_id"],
                        "pair_id": item["pair_id"],
                        "partition": partition,
                        "seed": item["seed"],
                        "condition": item["condition"],
                        "mechanism": item["mechanism"],
                        "factor_row": item["factor_row"],
                        "fault_schedule": to_jsonable(item["faults"]),
                        "success": False,
                        "reward": 0.0,
                        "terminated": True,
                        "truncated": False,
                        "termination_reason": "environment_done",
                    },
                    exposure=np.asarray([False]),
                )
                common = {
                    "episode_token": token,
                    "episode_id": item["episode_id"],
                    "pair_id": item["pair_id"],
                    "partition": partition,
                    "seed": item["seed"],
                    "feature_rows": 1,
                    "input_group": f"/episodes/{token}",
                    "label_group": f"/episodes/{token}",
                }
                index_records.append({"event_type": "dataset_episode", **common})
                episode_records.append(
                    {
                        "event_type": "episode",
                        **common,
                        "condition": item["condition"],
                        "mechanism": item["mechanism"],
                        "configured_fault_count": 0 if is_clean else 1,
                        "applied_fault_count": 0,
                        "exposed_rows": 0,
                        "not_exposed": not is_clean,
                        "success": False,
                        "steps": 1,
                        "reward": 0.0,
                        "termination_reason": "environment_done",
                        "action_saturated_values": 0,
                    }
                )
                audit_records.append(
                    {
                        "event_type": "audit_transition",
                        "episode_index": index,
                        "episode_id": item["episode_id"],
                        "episode_token": token,
                        "pair_id": item["pair_id"],
                        "condition": item["condition"],
                        "seed": item["seed"],
                        "control_step": 0,
                        "audit": {
                            "episode_id": index,
                            "step": 0,
                            "requested_action": action,
                            "executed_action": action,
                            "reward": 0.0,
                            "terminated": True,
                            "truncated": False,
                            "info": {"success": False},
                            "success": False,
                        },
                    }
                )

        _write_jsonl(directory / "dataset_index.jsonl", index_records)
        _write_jsonl(directory / "episodes.jsonl", episode_records)
        _write_jsonl(directory / "audit_stream.jsonl", audit_records)
        parameter_hash = "a" * 64
        policy_state = {
            "initial_parameter_sha256": parameter_hash,
            "current_parameter_sha256": parameter_hash,
            "model_training": False,
            "all_parameters_frozen": True,
        }
        _write_json(directory / "policy_state_before.json", policy_state)
        _write_json(directory / "policy_state_after.json", policy_state)
        software = {
            "packages": {"python": "3.11-test", "numpy": np.__version__},
            "gpu": [{"name": "test-gpu"}],
            "repository_commit": "1" * 40,
            "repository_dirty": False,
            "robocasa_commit": "2" * 40,
            "robosuite_commit": "3" * 40,
            "groot_commit": "4" * 40,
        }
        _write_json(directory / "software_versions.json", software)
        policy = {
            "name": self.policy_manifest["policy_name"],
            "checkpoint_sha256": self.policy_manifest["checkpoint_sha256"],
            "checkpoint_files_sha256": {
                record["path"]: record["sha256"]
                for record in self.policy_manifest["checkpoint_files"]
            },
            "parameter_sha256_before": parameter_hash,
            "parameter_sha256_after": parameter_hash,
            "frozen": True,
        }
        environment = {
            "id": self.protocol["environment"]["id"],
            "split": self.protocol["environment"]["split"],
            "horizon": self.protocol["environment"]["horizon"],
            "action_space": action_space,
        }
        artifact_paths = {
            "run_manifest": str(directory / "run_manifest.json"),
            "dataset_index": str(directory / "dataset_index.jsonl"),
            "monitor_inputs": str(directory / "monitor_inputs.h5"),
            "offline_labels": str(directory / "offline_labels.h5"),
            "episodes": str(directory / "episodes.jsonl"),
            "audit_stream": str(directory / "audit_stream.jsonl"),
            "metrics": str(directory / "metrics.json"),
            "software_versions": str(directory / "software_versions.json"),
            "policy_state_before": str(directory / "policy_state_before.json"),
            "policy_state_after": str(directory / "policy_state_after.json"),
            "artifact_validation": str(directory / "artifact_validation.json"),
            "shard_integrity": str(directory / "shard_integrity.json"),
        }
        metrics = {
            "status": "completed",
            "scientific_result": False,
            "debug": False,
            "collection_role": "formal_shard",
            "partition_complete": False,
            "protocol_version": self.protocol["protocol_version"],
            "monitor_protocol_version": self.protocol["monitor_protocol_version"],
            "partition": partition,
            "seeds": seeds,
            "episode_count": len(plan),
            "rows": len(plan),
            "conditions": {condition: len(seeds) for condition in CONDITIONS},
            "exposed_fault_episodes": 0,
            "not_exposed_fault_episodes": 2 * len(seeds),
            "feature_version": FEATURE_VERSION,
            "feature_count": len(FEATURE_NAMES),
            "environment": environment,
            "policy": policy,
            "outputs": artifact_paths,
        }
        _write_json(directory / "metrics.json", metrics)
        _write_json(
            directory / "run_manifest.json",
            {
                "protocol_version": self.protocol["protocol_version"],
                "monitor_protocol_version": self.protocol["monitor_protocol_version"],
                "status": "completed",
                "scientific_result": False,
                "debug": False,
                "collection_role": "formal_shard",
                "partition_complete": False,
                "partition": partition,
                "environment": environment,
                "policy": policy,
                "monitor_inputs": self.protocol["information_boundary"],
                "storage": self.protocol["storage"],
                "seeds": seeds,
                "episode_plan": plan,
                "config": {
                    "path": str(PROTOCOL_PATH.resolve()),
                    "sha256": protocol_sha256,
                },
                "policy_manifest": {
                    "path": str(POLICY_PATH.resolve()),
                    "sha256": policy_manifest_sha256,
                },
                "command": ["synthetic-test"],
                "artifacts": artifact_paths,
            },
        )
        _write_json(directory / "artifact_validation.json", {"status": "passed", "errors": []})
        _write_json(
            directory / "shard_integrity.json",
            build_shard_integrity_manifest(
                directory,
                partition=partition,
                collection_role="formal_shard",
                seeds=seeds,
                protocol_sha256=protocol_sha256,
                policy_manifest_sha256=policy_manifest_sha256,
            ),
        )

    def test_complete_disjoint_shards_pass_and_tampering_fails_closed(self) -> None:
        calibration = self.protocol["splits"]["calibration_scene_seeds"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "calibration-shard-00"
            second = root / "calibration-shard-01"
            self._write_formal_shard(first, partition="calibration", seeds=calibration[:25])
            self._write_formal_shard(second, partition="calibration", seeds=calibration[25:])

            complete = validate_formal_shard_set(
                PROTOCOL_PATH,
                POLICY_PATH,
                partition="calibration",
                shard_paths=[first, second],
            )
            self.assertTrue(complete["passed"], complete["errors"][:5])
            self.assertEqual(complete["observed_seed_count"], 50)
            self.assertEqual(complete["observed_episode_count"], 150)

            episode_path = first / "episodes.jsonl"
            original_episodes = episode_path.read_text(encoding="utf-8")
            episode_records = [json.loads(line) for line in original_episodes.splitlines()]
            episode_records[0]["action_saturated_values"] = -1
            _write_jsonl(episode_path, episode_records)
            _write_json(
                first / "shard_integrity.json",
                build_shard_integrity_manifest(
                    first,
                    partition="calibration",
                    collection_role="formal_shard",
                    seeds=calibration[:25],
                    protocol_sha256=sha256_file(PROTOCOL_PATH),
                    policy_manifest_sha256=sha256_file(POLICY_PATH),
                ),
            )
            semantically_tampered = validate_formal_shard_set(
                PROTOCOL_PATH,
                POLICY_PATH,
                partition="calibration",
                shard_paths=[first, second],
            )
            self.assertFalse(semantically_tampered["passed"])
            self.assertTrue(
                any(
                    "action_saturated_values is not non-negative" in error
                    for error in semantically_tampered["errors"]
                )
            )
            episode_path.write_text(original_episodes, encoding="utf-8")
            _write_json(
                first / "shard_integrity.json",
                build_shard_integrity_manifest(
                    first,
                    partition="calibration",
                    collection_role="formal_shard",
                    seeds=calibration[:25],
                    protocol_sha256=sha256_file(PROTOCOL_PATH),
                    policy_manifest_sha256=sha256_file(POLICY_PATH),
                ),
            )

            incomplete = validate_formal_shard_set(
                PROTOCOL_PATH,
                POLICY_PATH,
                partition="calibration",
                shard_paths=[first],
            )
            self.assertFalse(incomplete["passed"])
            self.assertEqual(incomplete["missing_seeds"], calibration[25:])

            duplicated = validate_formal_shard_set(
                PROTOCOL_PATH,
                POLICY_PATH,
                partition="calibration",
                shard_paths=[first, first, second],
            )
            self.assertFalse(duplicated["passed"])
            self.assertEqual(duplicated["duplicate_seeds"], calibration[:25])

            with episode_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            tampered = validate_formal_shard_set(
                PROTOCOL_PATH,
                POLICY_PATH,
                partition="calibration",
                shard_paths=[first, second],
            )
            self.assertFalse(tampered["passed"])
            self.assertTrue(
                any("episodes.jsonl.sha256 mismatch" in error for error in tampered["errors"])
            )

    def test_malformed_hdf5_is_reported_as_a_gate_failure(self) -> None:
        calibration = self.protocol["splits"]["calibration_scene_seeds"]
        with tempfile.TemporaryDirectory() as temporary:
            shard = Path(temporary) / "calibration-shard-00"
            seeds = calibration[:1]
            self._write_formal_shard(shard, partition="calibration", seeds=seeds)
            with h5py.File(shard / "monitor_inputs.h5", "r+") as stream:
                token = next(iter(stream["episodes"]))
                group = stream["episodes"][token]
                del group["features"]
                group.create_dataset("features", data=np.asarray(0.0, dtype=np.float32))
            _write_json(
                shard / "shard_integrity.json",
                build_shard_integrity_manifest(
                    shard,
                    partition="calibration",
                    collection_role="formal_shard",
                    seeds=seeds,
                    protocol_sha256=sha256_file(PROTOCOL_PATH),
                    policy_manifest_sha256=sha256_file(POLICY_PATH),
                ),
            )

            report = validate_formal_shard_set(
                PROTOCOL_PATH,
                POLICY_PATH,
                partition="calibration",
                shard_paths=[shard],
            )

            self.assertFalse(report["passed"])
            self.assertTrue(
                any("invalid monitor dataset" in error for error in report["errors"]),
                report["errors"],
            )


if __name__ == "__main__":
    unittest.main()
