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
from vla_recovery_bench.types import FaultPhase, FaultSpec


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


if __name__ == "__main__":
    unittest.main()
