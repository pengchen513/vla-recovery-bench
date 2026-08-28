import unittest

import numpy as np
from gymnasium import spaces

from vla_recovery_bench.policy_adapter import RoboCasaPolicyAdapter
from vla_recovery_bench.robocasa_adapter import (
    RandomPolicy,
    ZeroPolicy,
    action_shape,
    corrupt_image,
    validate_action,
    zero_action,
)


def _space() -> spaces.Dict:
    return spaces.Dict(
        {
            "action.base_motion": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "action.control_mode": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        }
    )


class RoboCasaActionContractTest(unittest.TestCase):
    def test_zero_action_preserves_dict_contract(self) -> None:
        action_space = _space()
        action = zero_action(action_space)
        validate_action(action_space, action)
        self.assertEqual(
            action_shape(action),
            {"action.base_motion": [4], "action.control_mode": [1]},
        )

    def test_zero_action_honors_discrete_space_starts(self) -> None:
        action_space = spaces.Tuple(
            (
                spaces.Discrete(3, start=5),
                spaces.MultiDiscrete([2, 3], start=[7, 11]),
            )
        )
        action = zero_action(action_space)
        validate_action(action_space, action)
        self.assertEqual(action[0], 5)
        np.testing.assert_array_equal(action[1], np.asarray([7, 11]))

    def test_invalid_structured_action_fails_closed(self) -> None:
        action_space = _space()
        with self.assertRaises(ValueError):
            validate_action(action_space, {"action.base_motion": np.zeros(4, dtype=np.float32)})
        with self.assertRaises(ValueError):
            validate_action(
                action_space,
                {
                    "action.base_motion": np.full(4, np.nan, dtype=np.float32),
                    "action.control_mode": np.zeros(1, dtype=np.float32),
                },
            )

    def test_policies_emit_legal_actions(self) -> None:
        action_space = _space()
        for policy in (ZeroPolicy(action_space), RandomPolicy(action_space)):
            action = policy.act({}, "test")
            validate_action(action_space, action)

    def test_policy_adapter_rejects_missing_observation_key(self) -> None:
        action_space = _space()
        adapter = RoboCasaPolicyAdapter(
            lambda observation, instruction: zero_action(action_space),
            action_space,
            {"state.pose": (3,)},
        )
        with self.assertRaises(ValueError):
            adapter.act({}, "test")

    def test_hard_image_corruptions_preserve_contract(self) -> None:
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        for variant in ("partial_mask", "blur", "color_shift", "all_zero"):
            corrupted = corrupt_image(image, variant)
            self.assertEqual(corrupted.shape, image.shape)
            self.assertEqual(corrupted.dtype, image.dtype)
        stale = corrupt_image(image, "stale_frame", stale_frame=np.zeros_like(image))
        self.assertTrue(np.all(stale == 0))

    def test_unknown_image_corruption_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            corrupt_image(np.zeros((4, 4, 3), dtype=np.uint8), "unknown")


if __name__ == "__main__":
    unittest.main()
