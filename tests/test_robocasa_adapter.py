import unittest

import numpy as np

from vla_recovery_bench.robocasa_adapter import find_rgb_observations


class RoboCasaAdapterTest(unittest.TestCase):
    def test_finds_nested_rgb_observation(self) -> None:
        observation = {"state": [0, 1], "camera": {"front": np.zeros((32, 48, 3))}}
        images = find_rgb_observations(observation)
        self.assertEqual(list(images), ["camera.front"])


if __name__ == "__main__":
    unittest.main()
