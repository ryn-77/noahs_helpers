from core.animal import Animal
from core.views.ark_view import ArkView


class Ark:
    def __init__(self, position: tuple[int, int]) -> None:
        self.position = position
        self.animals: set[Animal] = set()

    def get_view(self) -> ArkView:
        return ArkView(self.position, self.animals.copy())
