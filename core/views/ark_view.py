from dataclasses import dataclass

from core.animal import Animal


@dataclass(frozen=True)
class ArkView:
    position: tuple[int, int]
    animals: set[Animal]
