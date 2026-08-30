import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from gymnasium import spaces

from scripts.analyze_diagnostic_probe import _episode_loss, _load_jsonl
from scripts.lock_diagnostic_probe import _binomial_interval, _rate_report
from scripts.run_diagnostic_probe import (
    _action_delta,
    _observation_delta,
    _plan_for_stage,
    _posterior_delta,
    _validate_pair_prefixes,
    _validate_run_artifacts,
)
from vla_recovery_bench.diagnostic_probe import (
    assert_online_event_safe,
    choose_entropy_threshold,
    mechanism_log_loss,
    paired_cluster_bootstrap,
)
from vla_recovery_bench.groot_adapter import (
    ACTION_DIMS,
    ACTION_HORIZON,
    CAMERA_SHAPES,
    STATE_SHAPES,
    GrootRoboCasaPolicy,
)
from vla_recovery_bench.monitor import normalized_posterior_entropy
from vla_recovery_bench.monitor_protocol import validate_probe_protocol


def _action_space() -> spaces.Dict:
    return spaces.Dict(
        {
            key: spaces.Box(-1.0, 1.0, (width,), dtype=np.float32)
            for key, width in ACTION_DIMS.items()
        }
    )


def _observation() -> dict[str, object]:
    return {
        "video": {
            key.removeprefix("video."): np.full(shape, 100, dtype=np.uint8)
            for key, shape in CAMERA_SHAPES.items()
        },
        "state": {
            key.removeprefix("state."): np.zeros(shape, dtype=np.float32)
            for key, shape in STATE_SHAPES.items()
        },
        "annotation": {"human": {"task_description": np.asarray("put the cup away")}},
    }


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, endpoint: str, data: object = None) -> dict[str, object]:
        del endpoint, data
        self.calls += 1
        return {
            key: np.full((ACTION_HORIZON, width), 0.1, dtype=np.float32)
            for key, width in ACTION_DIMS.items()
        }


class DiagnosticProbeTest(unittest.TestCase):
    def test_pair_prefix_validator_requires_equal_prefixes(self) -> None:
        rows = [
            {
                "pair_id": "scene-1",
                "condition": "clean",
                "arm": "passive_only",
                "prefix_hash_to_trigger": "same",
                "prefix_step_count": 4,
                "trigger_observation_step": None,
            },
            {
                "pair_id": "scene-1",
                "condition": "clean",
                "arm": "passive_plus_probe",
                "prefix_hash_to_trigger": "same",
                "prefix_step_count": 4,
                "trigger_observation_step": None,
            },
        ]
        errors, report = _validate_pair_prefixes(rows)
        self.assertEqual(errors, [])
        self.assertTrue(report["passed"])
        rows[1]["prefix_hash_to_trigger"] = "different"
        errors, report = _validate_pair_prefixes(rows)
        self.assertTrue(any("hash mismatch" in error for error in errors))
        self.assertFalse(report["passed"])

    def test_pair_prefix_validator_rejects_missing_arm(self) -> None:
        errors, _ = _validate_pair_prefixes(
            [
                {
                    "pair_id": "scene-1",
                    "condition": "clean",
                    "arm": "passive_only",
                    "prefix_hash_to_trigger": "same",
                    "prefix_step_count": 4,
                    "trigger_observation_step": None,
                }
            ]
        )
        self.assertTrue(any("missing arms" in error for error in errors))

    def test_clopper_pearson_interval_is_report_only_and_well_formed(self) -> None:
        interval = _binomial_interval(0, 50)
        self.assertEqual(interval["lower"], 0.0)
        self.assertGreater(interval["upper"], 0.0)
        midpoint = _rate_report(2, 50, steps=37500)
        self.assertEqual(midpoint["episodes"], 2)
        self.assertEqual(midpoint["total_episodes"], 50)
        self.assertAlmostEqual(midpoint["rate"], 0.04)
        self.assertLessEqual(midpoint["clopper_pearson_95_percent"]["lower"], midpoint["rate"])
        self.assertGreaterEqual(midpoint["clopper_pearson_95_percent"]["upper"], midpoint["rate"])

    def test_entropy_extremes_and_order_statistic_lock(self) -> None:
        self.assertAlmostEqual(
            normalized_posterior_entropy(
                {"none": 1.0, "actuator_fault": 0.0, "observation_fault": 0.0}
            ),
            0.0,
        )
        self.assertAlmostEqual(
            normalized_posterior_entropy(
                {"none": 1 / 3, "actuator_fault": 1 / 3, "observation_fault": 1 / 3}
            ),
            1.0,
        )
        rows = [
            {"episode_id": str(index), "maximum_risk": 0.1, "maximum_entropy": value}
            for index, value in enumerate((0.1, 0.2, 0.3, 0.4))
        ]
        lock = choose_entropy_threshold(rows, risk_threshold=0.9, max_union_rate=0.5)
        self.assertEqual(lock["allowed_joint_trigger_episodes"], 2)
        self.assertLessEqual(lock["joint_trigger_rate"], 0.5)
        self.assertGreater(lock["entropy_threshold"], 0.2)

    def test_risk_alone_over_budget_fails_closed(self) -> None:
        rows = [
            {"maximum_risk": 1.0, "maximum_entropy": 0.1},
            {"maximum_risk": 1.0, "maximum_entropy": 0.1},
            {"maximum_risk": 1.0, "maximum_entropy": 0.1},
        ]
        with self.assertRaisesRegex(ValueError, "risk alarm alone"):
            choose_entropy_threshold(rows, risk_threshold=0.5, max_union_rate=0.5)

    def test_adapter_repeat_and_force_requery_preserve_episode_state(self) -> None:
        client = _Client()
        policy = GrootRoboCasaPolicy(_action_space(), client)
        first = policy.act(_observation(), "put the cup away")
        state_before = policy.chunk_state()
        repeated = policy.repeat_last_action()
        self.assertEqual(client.calls, 1)
        self.assertEqual(
            state_before["remaining_actions"], policy.chunk_state()["remaining_actions"]
        )
        np.testing.assert_array_equal(
            first["action.end_effector_position"], repeated["action.end_effector_position"]
        )
        policy.force_requery()
        self.assertEqual(policy.inference_count, 1)
        policy.act(_observation(), "put the cup away")
        self.assertEqual(client.calls, 2)
        self.assertEqual(policy.inference_count, 2)
        self.assertEqual(policy.chunk_state()["chunk_id"], state_before["chunk_id"] + 1)

    def test_online_firewall_and_offline_loss(self) -> None:
        assert_online_event_safe({"posterior": {"none": 0.5}, "requested_action": {}})
        with self.assertRaisesRegex(ValueError, "forbidden online probe field"):
            assert_online_event_safe({"audit": {"reward": 1.0}})
        with self.assertRaisesRegex(ValueError, "forbidden online probe field"):
            assert_online_event_safe({"metadata.reward": 1.0})

        @dataclass
        class Leaked:
            executed_action: object

        with self.assertRaisesRegex(ValueError, "forbidden online probe field"):
            assert_online_event_safe(Leaked(executed_action={}))
        self.assertAlmostEqual(
            mechanism_log_loss(
                {"none": 0.8, "actuator_fault": 0.1, "observation_fault": 0.1},
                "none",
            ),
            -np.log(0.8),
        )

    def test_trigger_recomputes_and_validates_entropy(self) -> None:
        from vla_recovery_bench.diagnostic_probe import trigger_from_prediction

        posterior = {"none": 0.8, "actuator_fault": 0.1, "observation_fault": 0.1}
        with self.assertRaisesRegex(ValueError, "does not match"):
            trigger_from_prediction(
                {"risk": 0.2, "posterior": posterior, "normalized_entropy": 0.0},
                risk_threshold=0.5,
                entropy_threshold=0.9,
            )
        with self.assertRaisesRegex(ValueError, "risk does not match"):
            trigger_from_prediction(
                {"risk": 0.3, "posterior": posterior},
                risk_threshold=0.5,
                entropy_threshold=0.9,
            )
        accepted = trigger_from_prediction(
            {"risk": 0.20000003, "posterior": posterior},
            risk_threshold=0.5,
            entropy_threshold=0.9,
        )
        self.assertFalse(accepted["joint_trigger"])

    def test_empty_probe_stream_is_valid_for_no_trigger_run(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probe_stream.jsonl"
            path.touch()
            self.assertEqual(_load_jsonl(path, allow_empty=True), [])
            with self.assertRaisesRegex(ValueError, "empty JSONL"):
                _load_jsonl(path)

    def test_probe_deltas_are_explicit_and_non_privileged(self) -> None:
        previous = {
            "video": {"cam": np.zeros((2, 2, 3), dtype=np.uint8)},
            "state": {"q": np.zeros(2)},
        }
        current = {
            "video": {"cam": np.full((2, 2, 3), 255, dtype=np.uint8)},
            "state": {"q": np.ones(2)},
        }
        deltas, camera = _observation_delta(previous, current)
        self.assertAlmostEqual(deltas["video.cam"]["mean_absolute"], 1.0)
        self.assertAlmostEqual(camera["temporal_consistency"], 0.0)
        action = {"action.x": np.ones(2, dtype=np.float32)}
        self.assertEqual(_action_delta(action, action)["action.x"]["mean_absolute"], 0.0)
        posterior = {"none": 0.8, "actuator_fault": 0.1, "observation_fault": 0.1}
        self.assertEqual(
            _posterior_delta({"posterior": posterior}, posterior),
            {"none": 0.0, "actuator_fault": 0.0, "observation_fault": 0.0},
        )

    def test_analyzer_no_trigger_is_zero_itt_and_bootstrap_is_deterministic(self) -> None:
        row = {"condition": "actuator_fault", "triggered": False}
        self.assertEqual(_episode_loss(row, "actuator_fault"), (0.0, "no_trigger_itt_zero"))
        values = [
            {"seed": 1, "paired_improvement_delta": 1.0},
            {"seed": 2, "paired_improvement_delta": 3.0},
        ]
        first = paired_cluster_bootstrap(values, replicates=100, seed=7)
        second = paired_cluster_bootstrap(values, replicates=100, seed=7)
        self.assertEqual(first, second)

    def test_v11_checked_in_protocol_passes(self) -> None:
        root = Path(__file__).parents[1]
        probe = json.loads((root / "configs/diagnostic_probe_v1_1.json").read_text())
        monitor = json.loads((root / "configs/monitor_training_v1_0.json").read_text())
        self.assertEqual(validate_probe_protocol(probe, monitor), [])

    def test_stage_plan_counts_and_final_is_not_implicitly_collectable(self) -> None:
        root = Path(__file__).parents[1]
        config = json.loads((root / "configs/identifiability_pilot_v1_4.json").read_text())
        probe = json.loads((root / "configs/diagnostic_probe_v1_1.json").read_text())
        debug_seeds, debug_plan = _plan_for_stage(config, "debug")
        pilot_seeds, pilot_plan = _plan_for_stage(config, "pilot")
        _, final_plan = _plan_for_stage(
            config, "final", final_seeds=probe["staging"]["final_seeds"]
        )
        self.assertEqual(len(debug_seeds), 3)
        self.assertEqual(len(debug_plan), 18)
        self.assertEqual(len(pilot_seeds), 12)
        self.assertEqual(len(pilot_plan), 72)
        self.assertEqual(final_plan, [])

    def test_artifact_gate_allows_shared_paired_episode_ids(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            for name in (
                "run_manifest.json",
                "metrics.json",
                "monitor_config.json",
                "calibration.json",
                "software_versions.json",
                "probe_lock.json",
                "policy_state_before.json",
                "policy_state_after.json",
                "privileged_audit.jsonl",
            ):
                (output / name).write_text("{}\n")
            (output / "probe_stream.jsonl").touch()
            episodes = [
                {
                    "episode_id": "scene-500-clean",
                    "episode_token": "token-passive",
                    "pair_id": "scene-500",
                    "arm": "passive_only",
                    "condition": "clean",
                    "seed": 500,
                    "probe_steps": 0,
                    "monitor_parameter_sha256": "hash",
                    "prefix_hash_to_trigger": "same-prefix",
                    "prefix_step_count": 0,
                    "trigger_observation_step": None,
                },
                {
                    "episode_id": "scene-500-clean",
                    "episode_token": "token-probe",
                    "pair_id": "scene-500",
                    "arm": "passive_plus_probe",
                    "condition": "clean",
                    "seed": 500,
                    "probe_steps": 0,
                    "monitor_parameter_sha256": "hash",
                    "prefix_hash_to_trigger": "same-prefix",
                    "prefix_step_count": 0,
                    "trigger_observation_step": None,
                },
            ]
            (output / "episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in episodes) + "\n"
            )
            (output / "monitor_stream.jsonl").write_text(
                '{"episode_token":"token-passive"}\n{"episode_token":"token-probe"}\n'
            )
            expected_plan = [
                {
                    **{key: row[key] for key in ("episode_id", "pair_id", "condition", "seed")},
                    "arm": row["arm"],
                }
                for row in episodes
            ]
            errors = _validate_run_artifacts(
                output,
                expected_count=2,
                stage="debug",
                expected_plan=expected_plan,
            )
            self.assertNotIn("duplicate episode id/arm pairs", errors)


if __name__ == "__main__":
    unittest.main()
