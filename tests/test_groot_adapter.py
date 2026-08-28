import unittest

import numpy as np
from gymnasium import spaces

from vla_recovery_bench.groot_adapter import (
    ACTION_DIMS,
    ACTION_HORIZON,
    CAMERA_SHAPES,
    PROMPT_KEY,
    STATE_SHAPES,
    GrootRoboCasaPolicy,
    flatten_observation,
    prepare_groot_observation,
    task_description,
)


def _observation() -> dict[str, object]:
    nested: dict[str, object] = {
        "video": {
            key.removeprefix("video."): np.zeros(shape, dtype=np.uint8)
            for key, shape in CAMERA_SHAPES.items()
        },
        "state": {
            key.removeprefix("state."): np.zeros(shape, dtype=np.float32)
            for key, shape in STATE_SHAPES.items()
        },
        "annotation": {"human": {"task_description": "put the cup away"}},
    }
    return nested


def _action_space() -> spaces.Dict:
    return spaces.Dict(
        {
            key: spaces.Box(-1.0, 1.0, shape=(width,), dtype=np.float32)
            for key, width in ACTION_DIMS.items()
        }
    )


class _FakeClient:
    def __init__(self, response: dict[str, np.ndarray]) -> None:
        self.response = response
        self.calls = 0

    def call(self, endpoint: str, data: object) -> dict[str, np.ndarray]:
        assert endpoint == "get_action"
        assert isinstance(data, dict)
        self.calls += 1
        return self.response


class GrootAdapterTest(unittest.TestCase):
    def test_recursive_keys_and_prompt_are_preserved(self) -> None:
        observation = _observation()
        flattened = flatten_observation(observation)
        expected = set(CAMERA_SHAPES) | set(STATE_SHAPES) | {PROMPT_KEY}
        self.assertEqual(set(flattened), expected)
        self.assertEqual(task_description(observation), "put the cup away")

        prepared = prepare_groot_observation(observation, "put the cup away")
        self.assertEqual(set(prepared), expected)
        self.assertEqual(prepared[PROMPT_KEY].shape, (1,))
        for key, shape in CAMERA_SHAPES.items():
            self.assertEqual(prepared[key].shape, (1, *shape))
        for key, shape in STATE_SHAPES.items():
            self.assertEqual(prepared[key].shape, (1, *shape))

    def test_policy_validates_and_queues_official_action_chunk(self) -> None:
        response = {
            key: np.zeros((ACTION_HORIZON, width), dtype=np.float32)
            for key, width in ACTION_DIMS.items()
        }
        client = _FakeClient(response)
        policy = GrootRoboCasaPolicy(_action_space(), client)

        first = policy.act(_observation(), "put the cup away")
        second = policy.act(_observation(), "put the cup away")

        self.assertEqual(set(first), set(ACTION_DIMS))
        self.assertEqual(set(second), set(ACTION_DIMS))
        self.assertEqual(client.calls, 1)
        self.assertEqual(policy.inference_count, 1)
        self.assertEqual(len(policy.requested_action_chunk), ACTION_HORIZON)

        exposed = policy.requested_action_chunk
        exposed[0]["action.base_motion"][0] = 1.0
        self.assertEqual(float(policy.requested_action_chunk[0]["action.base_motion"][0]), 0.0)

        policy.reset()
        self.assertEqual(policy.requested_action_chunk, ())

    def test_policy_rejects_nonfinite_action(self) -> None:
        response = {
            key: np.zeros((ACTION_HORIZON, width), dtype=np.float32)
            for key, width in ACTION_DIMS.items()
        }
        response["action.gripper_close"][0, 0] = np.nan
        policy = GrootRoboCasaPolicy(_action_space(), _FakeClient(response))

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            policy.act(_observation(), "put the cup away")

    def test_policy_audits_explicit_action_saturation(self) -> None:
        response = {
            key: np.zeros((ACTION_HORIZON, width), dtype=np.float32)
            for key, width in ACTION_DIMS.items()
        }
        response["action.end_effector_position"][0, 0] = 1.25
        policy = GrootRoboCasaPolicy(_action_space(), _FakeClient(response))

        action = policy.act(_observation(), "put the cup away")

        self.assertEqual(float(action["action.end_effector_position"][0]), 1.0)
        self.assertEqual(policy.episode_saturated_values, 1)
        assert policy.last_chunk_saturation is not None
        self.assertEqual(policy.last_chunk_saturation["saturated_values"], 1)

    def test_adapter_rejects_silent_observation_drop(self) -> None:
        observation = _observation()
        del observation["video"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "missing="):
            prepare_groot_observation(observation, "put the cup away")


if __name__ == "__main__":
    unittest.main()
