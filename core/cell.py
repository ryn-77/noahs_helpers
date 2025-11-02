from __future__ import annotations

from core.animal import Animal
from core.views.cell_view import CellView
from core.views.player_view import PlayerView


class Cell:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.animals: set[Animal] = set()
        self.helpers: set[PlayerView] = set()

        self.up: Cell | None = None
        self.down: Cell | None = None
        self.left: Cell | None = None
        self.right: Cell | None = None

    def get_view(self, make_unknown: bool) -> CellView:
        return CellView(
            self.x,
            self.y,
            {animal.copy(make_unknown) for animal in self.animals},
            self.helpers,
        )
