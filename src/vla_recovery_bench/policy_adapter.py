"""Strict runtime boundary between RoboCasa observations and a frozen policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .robocasa_adapter import validate_action
from .types import Action, Observation


class RoboCasaPolicyAdapter:
    """Validate a frozen predictor without changing its observation contract.

    ``predict`` receives the original nested observation mapping. Any image or
    state preprocessing must be implemented explicitly by the checkpoint
    adapter and documented in its manifest; this wrapper never flattens or
    silently drops fields.
    """

    def __init__(
        self,
        predict: Callable[[Observation, str], Action],
        action_space: Any,
        observation_shapes: Mapping[str, tuple[int, ...]],
    ) -> None:
        self._predict = predict
        self._action_space = action_space
        self._observation_shapes = dict(observation_shapes)

    def reset(self) -> None:
        reset = getattr(self._predict, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: Observation, instruction: str) -> Action:
        self._validate_observation(observation)
        action = self._predict(observation, instruction)
        validate_action(self._action_space, action)
        return action

    def _validate_observation(self, observation: Observation) -> None:
        actual: dict[str, Any] = {}

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
                return
            actual[prefix] = value

        visit("", observation)
        if set(actual) != set(self._observation_shapes):
            raise ValueError(
                "observation keys do not match policy contract: "
                f"expected={sorted(self._observation_shapes)}, got={sorted(actual)}"
            )
        for key, expected_shape in self._observation_shapes.items():
            shape = tuple(int(dimension) for dimension in getattr(actual[key], "shape", ()))
            if shape != tuple(expected_shape):
                raise ValueError(
                    f"observation shape mismatch for {key}: expected={expected_shape}, got={shape}"
                )
