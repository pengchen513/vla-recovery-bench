from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class JsonlRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")

    def record(self, event_type: str, **payload: Any) -> None:
        event = {"event_type": event_type, **payload}
        self._stream.write(json.dumps(to_jsonable(event), sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> JsonlRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
