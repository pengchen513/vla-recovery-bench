import unittest

from vla_recovery_bench.faults import FaultSchedule
from vla_recovery_bench.types import FaultPhase, FaultSpec


class FaultScheduleTest(unittest.TestCase):
    def test_fault_is_emitted_once_per_reset(self) -> None:
        fault = FaultSpec("f1", "object_displacement", 2, FaultPhase.AFTER_STEP)
        schedule = FaultSchedule([fault])

        self.assertEqual(schedule.due(2, FaultPhase.AFTER_STEP), (fault,))
        self.assertEqual(schedule.due(2, FaultPhase.AFTER_STEP), ())
        schedule.reset()
        self.assertEqual(schedule.due(2, FaultPhase.AFTER_STEP), (fault,))

    def test_duplicate_fault_ids_are_rejected(self) -> None:
        fault = FaultSpec("f1", "x", 0, FaultPhase.BEFORE_ACTION)
        with self.assertRaises(ValueError):
            FaultSchedule([fault, fault])


if __name__ == "__main__":
    unittest.main()

