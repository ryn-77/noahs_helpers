from core.ark import Ark
from core.message import Message
from core.player import Player
from core.cell import Cell
from core.sight import Sight
from core.snapshots import HelperSurroundingsSnapshot

import core.constants as c


class Engine:
    def __init__(
        self,
        grid: list[list[Cell]],
        ark: Ark,
        helpers: list[Player],
        time: int,
    ) -> None:
        self.grid = grid
        self.ark = ark
        self.helpers = helpers
        self.time = time

    def _get_sights(self) -> dict[Player, list[Player]]:
        in_sight: dict[Player, list[Player]] = {helper: [] for helper in self.helpers}

        for i, helper in enumerate(self.helpers):
            for j in range(i + 1, len(self.helpers)):
                neighbor = self.helpers[j]
                if helper.distance(neighbor) <= c.MAX_SIGHT_KM:
                    in_sight[helper].append(neighbor)
                    in_sight[neighbor].append(helper)

        return in_sight

    def run_simulation(self) -> None:
        for time_elapsed in range(self.time):
            is_raining = time_elapsed >= self.time - c.START_RAIN
            ark_view = self.ark.get_view()

            print(f"time_elapsed: {time_elapsed}, is_raining: {is_raining}")

            # 1. show helpers their new surroundings:
            # a their position
            # b animals and helpers within 5km sight
            # c turn number
            # d whether it is raining
            # e their flock
            # f if they're in the Ark cell, the current view of the Ark BEFORE unloading any helpers' flock currently in the cell into it.

            # 2. get helpers' one byte message:

            sights = self._get_sights()
            messages_to: dict[Player, list[Message]] = {
                helper: [] for helper in self.helpers
            }

            for helper in self.helpers:
                sight = Sight(helper.position, self.grid)

                helper_ark_view = None
                if helper.is_in_ark():
                    helper_ark_view = ark_view

                snapshot = HelperSurroundingsSnapshot(
                    time_elapsed, is_raining, helper.position, sight, helper_ark_view
                )
                one_byte_message = helper.check_surroundings(snapshot)
                if not (0 <= one_byte_message <= c.ONE_BYTE):
                    raise Exception(
                        f"helper {helper.id} gave incorrect message: {one_byte_message}"
                    )

                # broadcast message to all neighbors
                for neighbor in sights[helper]:
                    msg = Message(helper.get_view(), one_byte_message)
                    messages_to[neighbor].append(msg)

            # 3. broadcast helpers' one byte message to all other helpers in their sight.

            # 4. Let helpers take action on their surroundings:
            # a obtain and/or release animals in their sight
            # b move in any direction

            for helper in self.helpers:
                action = helper.get_action(messages_to[helper])

            # 5. let free animals move with 0.5 probability
