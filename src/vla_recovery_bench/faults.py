from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .types import FaultPhase, FaultSpec


class FaultSchedule:
    """A deterministic one-shot fault schedule for a single episode."""

    def __init__(self, faults: Iterable[FaultSpec]) -> None:
        self._faults = tuple(faults)
        ids = [fault.fault_id for fault in self._faults]
        if len(ids) != len(set(ids)):
            raise ValueError("fault_id values must be unique")
        self._by_key: dict[tuple[int, FaultPhase], list[FaultSpec]] = defaultdict(list)
        for fault in self._faults:
            self._by_key[(fault.step, fault.phase)].append(fault)
        self.reset()

    @property
    def faults(self) -> tuple[FaultSpec, ...]:
        return self._faults

    def reset(self) -> None:
        self._consumed: set[str] = set()

    def due(self, step: int, phase: FaultPhase) -> tuple[FaultSpec, ...]:
        due_faults = tuple(
            fault
            for fault in self._by_key.get((step, phase), ())
            if fault.fault_id not in self._consumed
        )
        self._consumed.update(fault.fault_id for fault in due_faults)
        return due_faults


def fault_specs_from_config(raw_faults: Iterable[dict[str, object]]) -> list[FaultSpec]:
    specs = []
    for raw in raw_faults:
        specs.append(
            FaultSpec(
                fault_id=str(raw["fault_id"]),
                kind=str(raw["kind"]),
                step=int(raw["step"]),
                phase=FaultPhase(str(raw["phase"])),
                parameters=dict(raw.get("parameters", {})),
            )
        )
    return specs

