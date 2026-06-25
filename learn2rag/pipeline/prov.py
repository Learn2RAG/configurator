from dataclasses import dataclass, field
from secrets import token_hex
from typing import Any


@dataclass
class Activity():
    label: str
    startedAtTime: float
    endedAtTime: float
    data: Any = field(default_factory=dict)


class Prov():
    def __init__(self) -> None:
        self.id = token_hex()
        self.items: list[Activity] = []

    def append(self, activity: Activity) -> None:
        self.items.append(activity)
