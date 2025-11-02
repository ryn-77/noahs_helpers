from core.message import Message
from core.player import Player
from core.snapshots import HelperSurroundingsSnapshot


class RandomPlayer(Player):
    def __init__(self, id: int, ark_x: int, ark_y: int):
        super().__init__(id, ark_x, ark_y)
        print(f"I am {self}")

    def run(self) -> None:
        pass

    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot):
        return 0

    def get_action(self, one_byte_messages: list[Message]) -> int:
        return 0
