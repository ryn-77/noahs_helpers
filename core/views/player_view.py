from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerView:
    id: int
