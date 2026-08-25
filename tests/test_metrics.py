import unittest

from vla_recovery_bench.metrics import aggregate_episode_metrics
from vla_recovery_bench.types import EpisodeResult


class MetricsTest(unittest.TestCase):
    def test_recovery_and_detection_metrics(self) -> None:
        results = [
            EpisodeResult(0, 0, True, 8, 1.0, 2, 2, 0, (0, 1), True, True, "success"),
            EpisodeResult(1, 1, False, 10, 0.0, 2, 1, 1, (2,), True, False, "horizon"),
        ]

        metrics = aggregate_episode_metrics(results)

        self.assertEqual(metrics["success_rate"], 0.5)
        self.assertEqual(metrics["detection_precision"], 0.75)
        self.assertEqual(metrics["detection_recall"], 0.75)
        self.assertEqual(metrics["recovery_success_rate"], 0.5)
        self.assertEqual(metrics["mean_detection_delay_steps"], 1.0)


if __name__ == "__main__":
    unittest.main()

