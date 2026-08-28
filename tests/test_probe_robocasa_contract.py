import unittest

import numpy as np
from gymnasium import spaces

from scripts.probe_robocasa_contract import (
    ZeroPolicy,
    describe_observation,
    find_contract_images,
    validate_action,
)


class ContractProbeHelpersTest(unittest.TestCase):
    def test_describe_observation_recurses_all_leaf_keys(self) -> None:
        observation = {
            "state": {"pose": np.zeros(3, dtype=np.float32)},
            "camera": {"front": np.ones((4, 5, 3), dtype=np.uint8)},
        }
        records = describe_observation(observation)
        self.assertEqual([record["key"] for record in records], ["state.pose", "camera.front"])
        self.assertEqual(records[0]["shape"], [3])
        self.assertEqual(records[0]["dtype"], "float32")
        self.assertEqual(records[1]["minimum"], 1)
        self.assertEqual(records[1]["maximum"], 1)

    def test_find_contract_images_accepts_rgb_and_rgba_only(self) -> None:
        observation = {
            "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "rgba": np.zeros((2, 3, 4), dtype=np.uint8),
            "gray": np.zeros((2, 3), dtype=np.uint8),
            "nested": {"not_image": np.zeros((2, 3, 2), dtype=np.uint8)},
        }
        images = find_contract_images(observation)
        self.assertEqual(set(images), {"rgb", "rgba"})

    def test_zero_policy_honors_discrete_space_starts(self) -> None:
        action_space = spaces.Tuple(
            (
                spaces.Discrete(3, start=5),
                spaces.MultiDiscrete([2, 3], start=[7, 11]),
            )
        )
        action = ZeroPolicy()(action_space)
        validate_action(action_space, action)


if __name__ == "__main__":
    unittest.main()
