from core.player import Player


class Player7(Player):
    def __init__(self, id: int, ark_x: int, ark_y: int):
        super().__init__(id, ark_x, ark_y)

    def run(self) -> None:
        pass
