import json
import tempfile
import unittest
from pathlib import Path

from vla_recovery_bench.dummy import (
    ConstantFrozenPolicy,
    DummyProgressEnvironment,
    ProgressFailureMonitor,
    RetryRecoveryController,
)
from vla_recovery_bench.faults import FaultSchedule
from vla_recovery_bench.recording import JsonlRecorder
from vla_recovery_bench.runner import ExperimentRunner, RunnerConfig
from vla_recovery_bench.types import (
    FaultPhase,
    FaultSpec,
    MonitorContext,
    RecoveryAction,
    RecoveryDecision,
)


class RunnerTest(unittest.TestCase):
    def test_runner_detects_fault_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            environment = DummyProgressEnvironment(goal_progress=6)
            with JsonlRecorder(event_path) as recorder:
                runner = ExperimentRunner(
                    environment=environment,
                    policy=ConstantFrozenPolicy(),
                    monitor=ProgressFailureMonitor(stagnation_patience=1),
                    recovery=RetryRecoveryController(max_retries=2),
                    fault_schedule=FaultSchedule(
                        [
                            FaultSpec(
                                "shift",
                                "object_displacement",
                                2,
                                FaultPhase.AFTER_STEP,
                                {"magnitude": 2},
                            )
                        ]
                    ),
                    recorder=recorder,
                    config=RunnerConfig(1, 10, 11, "finish"),
                )
                results = runner.run()

            self.assertTrue(environment.closed)
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].fault_count, 1)
            self.assertEqual(results[0].detected_fault_count, 1)
            self.assertTrue(results[0].recovery_success)
            events = [json.loads(line) for line in event_path.read_text().splitlines()]
            self.assertEqual(events[0]["event_type"], "experiment_start")
            self.assertEqual(events[-1]["event_type"], "experiment_end")
            fault_events = [event for event in events if event["event_type"] == "fault_injection"]
            self.assertEqual(len(fault_events), 1)
            self.assertTrue(fault_events[0]["application"]["applied"])

    def test_recorder_refuses_non_empty_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                JsonlRecorder(path)

    def test_monitor_context_has_no_reward_or_info_and_audit_keeps_outcome(self) -> None:
        captured = []

        class CaptureMonitor(ProgressFailureMonitor):
            def observe(self, context):
                captured.append(context)
                return super().observe(context)

        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            environment = DummyProgressEnvironment(goal_progress=2)
            with JsonlRecorder(event_path) as recorder:
                ExperimentRunner(
                    environment=environment,
                    policy=ConstantFrozenPolicy(),
                    monitor=CaptureMonitor(stagnation_patience=1),
                    recovery=RetryRecoveryController(max_retries=0),
                    fault_schedule=FaultSchedule([]),
                    recorder=recorder,
                    config=RunnerConfig(1, 3, 0, "finish"),
                ).run()

            self.assertTrue(captured)
            self.assertFalse(hasattr(captured[0], "reward"))
            self.assertFalse(hasattr(captured[0], "info"))
            events = [json.loads(line) for line in event_path.read_text().splitlines()]
            audit = [event for event in events if event["event_type"] == "audit_transition"]
            self.assertTrue(audit)
            self.assertIn("reward", audit[0]["audit"])
            self.assertIn("executed_action", audit[0]["audit"])

    def test_monitor_context_rejects_privileged_observation_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden monitor field"):
            MonitorContext(
                episode_id=0,
                step=0,
                instruction="test",
                previous_observation={"state": {"pose": [0.0]}},
                observation={"info": {"success": False}},
                action=[0.0],
            )

    def test_action_chunk_reset_and_schedule_factory_are_audited(self) -> None:
        class ChunkPolicy(ConstantFrozenPolicy):
            action_chunk_length = 2

        class RequeryMonitor(ProgressFailureMonitor):
            def observe(self, context):
                decision = super().observe(context)
                if context.step == 0:
                    return decision
                return decision

        class RequeryController(RetryRecoveryController):
            def decide(self, context):
                if context.step == 1:
                    return RecoveryDecision(RecoveryAction.REQUERY_POLICY, "test requery")
                return RecoveryDecision(RecoveryAction.CONTINUE, "continue")

        schedules = []

        def factory(episode_id, seed):
            schedule = FaultSchedule(
                [FaultSpec(f"f-{seed}", "actuator_dropout", 0, FaultPhase.BEFORE_ACTION)]
            )
            schedules.append((episode_id, seed, schedule))
            return schedule

        with tempfile.TemporaryDirectory() as directory:
            with JsonlRecorder(Path(directory) / "events.jsonl") as recorder:
                ExperimentRunner(
                    environment=DummyProgressEnvironment(goal_progress=2),
                    policy=ChunkPolicy(),
                    monitor=RequeryMonitor(stagnation_patience=99),
                    recovery=RequeryController(),
                    fault_schedule=FaultSchedule([]),
                    fault_schedule_factory=factory,
                    recorder=recorder,
                    config=RunnerConfig(1, 3, 7, "finish"),
                ).run()
            self.assertEqual([(0, 7)], [(episode_id, seed) for episode_id, seed, _ in schedules])
            event_lines = (Path(directory) / "events.jsonl").read_text().splitlines()
            events = [json.loads(line) for line in event_lines]
            step_events = [event for event in events if event["event_type"] == "step"]
            self.assertEqual(step_events[2]["chunk"]["chunk_id"], 1)
            self.assertEqual(step_events[2]["chunk"]["position_in_chunk"], 0)


if __name__ == "__main__":
    unittest.main()
