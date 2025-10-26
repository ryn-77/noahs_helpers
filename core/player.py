from abc import ABC, abstractmethod

from core.animal import Animal


class Player(ABC):
    def __init__(self, id: int, ark_x: int, ark_y: int):
        self.id = id
        self.ark_position = (ark_x, ark_y)
        self.position = (float(ark_x), float(ark_y))
        self.flock: list[Animal] = []

    def __str__(self) -> str:
        return f"{self.__module__}(id={self.id})"

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def run(self) -> None:
        pass
