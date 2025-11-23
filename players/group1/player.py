from random import random
import math
from math import atan2, pi

from core.action import Action, Move, Obtain
from core.message import Message
from core.player import Player
from core.snapshots import HelperSurroundingsSnapshot
from core.views.player_view import Kind
from core.views.cell_view import CellView
from core.animal import Animal, Gender
from core import constants


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return (abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5


class Player1(Player):
    def __init__(
        self,
        id: int,
        ark_x: int,
        ark_y: int,
        kind: Kind,
        num_helpers: int,
        species_populations: dict[str, int],
    ):
        super().__init__(id, ark_x, ark_y, kind, num_helpers, species_populations)
        # print(f"I am {self}")

        self.is_raining = False
        self.escaping_wall: bool = False
        self.hellos_received = []
        self.chase_too_long_count = []
        self.ark_memory = {}
        self.memory_timestamp = 0
        self.last_ark_visit_turn = -1
        self.current_turn = 0
        self.seen_animals: set[int] = set()
        self.known_pairs: set[tuple[str, Gender]] = set()
        self.target_animal: Animal | None = None
        self.turns_chasing_target = 0
        self.max_chase_turns = 5

        # Corner mission state
        self.corner_mission_active = False
        self.corner_mission_done = False  # once we've reached the corner, this stays True
        self.corner_cells: list[tuple[int, int]] = []
        self.corner_idx: int = 0

        # --- Corner exploration mission setup ---

        # List of all 4 corner cells
        if self.kind != Kind.Noah and self.num_helpers > 0:
            all_corners: list[tuple[int, int]] = [
                (0, 0),
                (0, 999),
                (999, 0),
                (999, 999),
            ]

            num_corner_helpers = min(4, self.num_helpers)

            # Make helper indices 0..num_helpers-1 instead of using the raw id
            helper_index = (self.id - 1) % self.num_helpers

            if helper_index < num_corner_helpers:
                cx, cy = all_corners[helper_index]
                self.corner_cells = [(cx, cy)]
                self.corner_idx = 0
                self.corner_mission_active = True
                # print(f"Helper {self.id} assigned corner {cx, cy}")

    def _get_my_cell(self) -> CellView:
        xcell, ycell = tuple(map(int, self.position))
        if not self.sight.cell_is_in_sight(xcell, ycell):
            raise Exception(f"{self} failed to find own cell")

        return self.sight.get_cellview_at(xcell, ycell)

    def _do_corner_mission_move(self) -> Move | None:
        if not self.corner_mission_active:
            return None

        if self.corner_idx >= len(self.corner_cells):
            self.corner_mission_active = False
            return None

        target_cell_x, target_cell_y = self.corner_cells[self.corner_idx]
        cur_cell_x, cur_cell_y = int(self.position[0]), int(self.position[1])

        # If we've reached the target corner cell
        if (cur_cell_x, cur_cell_y) == (target_cell_x, target_cell_y):
            self.corner_idx += 1

            # Mission finished: behave like a normal helper starting from the corner
            if self.corner_idx >= len(self.corner_cells):
                self.corner_mission_active = False
                self.corner_mission_done = True  # permanently mark as "corner-started"
                return None

            # Otherwise, set next corner target (if you ever add more than one)
            target_cell_x, target_cell_y = self.corner_cells[self.corner_idx]

        # Move toward center of current target cell
        tx, ty = target_cell_x + 0.5, target_cell_y + 0.5
        return Move(*self.move_towards(tx, ty))

    def _update_ark_memory(self):
        self.ark_memory = {}
        if self.ark_view is None:
            return
        for animal in self.ark_view.animals:
            species, gender = animal.species_id, animal.gender
            if species not in self.ark_memory:
                self.ark_memory[species] = set()
            self.ark_memory[species].add(gender)

        self.memory_timestamp = self.current_turn
        self.last_ark_visit_turn = self.current_turn

    def _is_animal_needed(self, species, gender) -> bool:
        for animals in self.ark_memory:
            if animals == species and gender in self.ark_memory[animals]:
                # print(f"{self.id} does not need {animals} SHOULD MOVE ON")
                return False
        return True

    def _already_have_in_flock(self, species, gender) -> bool:
        for animal in self.flock:
            if animal.species_id == species and animal.gender == gender:
                # print(f"{self.id} already has {animal} in flock SHOULD MOVE ON")
                return True
        return False

    def _encode_memory_message(self) -> int:
        species_list = sorted(self.species_populations.keys())
        if len(species_list) == 0:
            return 0

        species_idx = self.current_turn % len(species_list)
        species = species_list[species_idx]

        msg = species_idx & 0x0F

        if species in self.ark_memory:
            genders = self.ark_memory[species]
            if Gender.Male in genders:
                msg |= 0x10
            if Gender.Female in genders:
                msg |= 0x20

        time_bits = (self.memory_timestamp % 4) << 6
        msg |= time_bits

        return msg & 0xFF

    def _decode_memory_message(self, msg: int) -> tuple[int, set, int]:
        species_idx = msg & 0x0F
        genders = set()
        if msg & 0x10:
            genders.add(Gender.Male)
        if msg & 0x20:
            genders.add(Gender.Female)
        timestamp = (msg >> 6) & 0x03
        return species_idx, genders, timestamp

    def _update_from_message(self, msg_content: int, msg_timestamp_hint: int):
        species_idx, genders, msg_time_bits = self._decode_memory_message(msg_content)
        species_list = sorted(self.species_populations.keys())

        if species_idx >= len(species_list):
            return

        species = species_list[species_idx]

        inferred_timestamp = (msg_timestamp_hint // 4) * 4 + msg_time_bits
        if msg_timestamp_hint % 4 > msg_time_bits and msg_timestamp_hint >= 4:
            inferred_timestamp += 4

        if inferred_timestamp > self.memory_timestamp:
            if species not in self.ark_memory:
                self.ark_memory[species] = set()
            self.ark_memory[species].update(genders)
            self.memory_timestamp = inferred_timestamp

    def _find_best_animal(self) -> Animal | None:
        closest_animal = None
        closest_dist = float("inf")

        for cellview in self.sight:
            for animal in cellview.animals:
                if animal in self.flock:
                    continue

                if id(animal) in self.seen_animals:
                    continue

                if not self._is_animal_needed(animal.species_id, animal.gender):
                    continue

                if self._already_have_in_flock(animal.species_id, animal.gender):
                    continue

                if animal.gender == Gender.Unknown:
                    if animal.species_id in self.ark_memory:
                        continue

                dist = distance(*self.position, cellview.x, cellview.y)
                if closest_animal is None or dist < closest_dist:
                    closest_animal = animal
                    closest_dist = dist

        return closest_animal

    def _find_animal_position(self, animal: Animal) -> tuple[float, float] | None:
        for cellview in self.sight:
            if animal in cellview.animals:
                return (cellview.x + 0.5, cellview.y + 0.5)
        return None

    def _is_near_wall(self, margin: float = 10.0) -> bool:
        x, y = self.position
        return (
            x <= margin or y <= margin or x >= 1000.0 - margin or y >= 1000.0 - margin
        )

    def _get_inward_from_wall_target(self, margin: float = 20.0) -> tuple[float, float]:
        """
        For corner helpers (corner_mission_done=True), when near the wall,
        move them a bit inward into the map instead of back to the ark.
        """
        x, y = self.position
        target_x, target_y = x, y

        if x <= margin:
            target_x = margin + 20.0
        elif x >= 1000.0 - margin:
            target_x = 1000.0 - margin - 20.0

        if y <= margin:
            target_y = margin + 20.0
        elif y >= 1000.0 - margin:
            target_y = 1000.0 - margin - 20.0

        target_x = min(max(target_x, 0.0), 999.9)
        target_y = min(max(target_y, 0.0), 999.9)
        return target_x, target_y

    def _is_safely_away_from_wall(self, safe_margin: float = 100.0) -> bool:
        x, y = self.position
        return (
            safe_margin <= x <= 1000.0 - safe_margin
            and safe_margin <= y <= 1000.0 - safe_margin
        )

    def _get_away_from_wall_move(self) -> tuple[float, float]:
        target_x, target_y = self.ark_position
        if self.can_move_to(target_x, target_y):
            return target_x, target_y
        return self.position[0] + 1, self.position[1] + 1

    def _get_random_move(self) -> tuple[float, float]:
        old_x, old_y = self.position
        dx, dy = random() - 0.5, random() - 0.5

        while not (self.can_move_to(old_x + dx, old_y + dy)):
            dx, dy = random() - 0.5, random() - 0.5

        return old_x + dx, old_y + dy

    def _get_move_direction(self) -> tuple[float, float]:
        # Special exploration direction for helpers that started in a corner:
        # push them diagonally inward instead of using the ark-based spread.
        if self.corner_mission_done and self.corner_cells:
            cx, cy = self.corner_cells[0]
            inv_sqrt2 = math.sqrt(0.5)

            if cx == 0 and cy == 0:
                # top-left corner -> down-right
                self.base_explore_dir = (inv_sqrt2, inv_sqrt2)
            elif cx == 0 and cy == 999:
                # bottom-left corner -> up-right
                self.base_explore_dir = (inv_sqrt2, -inv_sqrt2)
            elif cx == 999 and cy == 0:
                # top-right corner -> down-left
                self.base_explore_dir = (-inv_sqrt2, inv_sqrt2)
            elif cx == 999 and cy == 999:
                # bottom-right corner -> up-left
                self.base_explore_dir = (-inv_sqrt2, -inv_sqrt2)
            else:
                # Fallback, should not happen with pure corners
                self.base_explore_dir = (inv_sqrt2, inv_sqrt2)

            return self.base_explore_dir

        # Original logic for non-corner helpers
        if self.kind == Kind.Noah or self.num_helpers <= 0:
            self.base_explore_dir = (0.0, 0.0)
            return self.base_explore_dir

        ark_x, ark_y = self.ark_position
        map_center = 1000 / 2.0

        if ark_x == map_center and ark_y == map_center:
            angle = 2.0 * pi * (self.id / self.num_helpers)
            self.base_explore_dir = (math.cos(angle), math.sin(angle))
            return self.base_explore_dir

        dist_to_left = ark_x
        dist_to_right = 1000 - ark_x
        dist_to_bottom = ark_y
        dist_to_top = 1000 - ark_y

        space_scores = {
            (1.0, 1.0): dist_to_right + dist_to_top,
            (1.0, -1.0): dist_to_right + dist_to_bottom,
            (-1.0, 1.0): dist_to_left + dist_to_top,
            (-1.0, -1.0): dist_to_left + dist_to_bottom,
        }
        sorted_spaces = sorted(
            space_scores.items(), key=lambda item: item[1], reverse=True
        )

        best_quadrant = sorted_spaces[0][0]

        idx = self.id % self.num_helpers
        # if sorted_spaces[0][1] < 1.5 * sorted_spaces[1][1]:
        #     if idx > self.num_helpers - 2:
        #         best_quadrant = second_best_quadrant

        base_angle = atan2(best_quadrant[1], best_quadrant[0])
        spread = (pi / 1.5) * (idx / max(self.num_helpers - 1, 1)) - (pi / 4.0)
        final_angle = base_angle + spread

        self.base_explore_dir = (math.cos(final_angle), math.sin(final_angle))
        return self.base_explore_dir

    def _get_angled_move(self) -> tuple[float, float]:
        dir_x, dir_y = self._get_move_direction()

        if (dir_x, dir_y) == (0.0, 0.0):
            return self._get_random_move()

        old_x, old_y = self.position
        step = 1.0
        target_x = old_x + dir_x * step
        target_y = old_y + dir_y * step

        target_x = min(max(target_x, 0.0), 999.9)
        target_y = min(max(target_y, 0.0), 999.9)

        if self.can_move_to(target_x, target_y):
            return target_x, target_y

        return self._get_random_move()

    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot) -> int:
        self.position = snapshot.position
        self.flock = snapshot.flock

        for a in self.flock:
            self.seen_animals.add(id(a))

        self.sight = snapshot.sight
        self.is_raining = snapshot.is_raining
        self.ark_view = snapshot.ark_view

        self.current_turn += 1

        if self.is_in_ark():
            self._update_ark_memory()

        msg = self._encode_memory_message()

        if not self.is_message_valid(msg):
            msg = msg & 0xFF

        return msg

    def get_action(self, messages: list[Message]) -> Action | None:
        # Process messages
        for msg in messages:
            self._update_from_message(msg.contents, self.current_turn)

        # Noah never moves
        if self.kind == Kind.Noah:
            return None

        # If we're on the ark, cancel any wall-escape state
        if self.position == self.ark_position:
            self.escaping_wall = False

        # Rain emergency overrides everything
        if self.is_raining:
            return Move(*self.move_towards(*self.ark_position))

        # --- Corner mission movement has highest priority ---
        if self.corner_mission_active:
            corner_move = self._do_corner_mission_move()
            if corner_move is not None:
                return corner_move
            # If mission finished inside _do_corner_mission_move(),
            # corner_mission_active will be False and corner_mission_done may now be True.

        # --- Wall handling ---
        if self.corner_mission_done:
            # Special rule ONLY for corner helpers: stay away from wall, but don't go to ark
            if self._is_near_wall():
                tx, ty = self._get_inward_from_wall_target()
                return Move(*self.move_towards(tx, ty))
        else:
            # ORIGINAL wall-escape behavior for non-corner helpers
            if self.escaping_wall:
                return Move(*self.move_towards(*self.ark_position))

            if self._is_near_wall():
                # print(f"{self.id} near wall, initiating escape")
                self.escaping_wall = True
                return Move(*self.move_towards(*self.ark_position))

        # If we weren't near the wall (or are a corner helper away from wall), reset escape state
        if not self._is_near_wall():
            self.escaping_wall = False

        # ------------------------------------------------------------------
        # From here on: normal behavior (chasing / obtaining / exploring)
        # ------------------------------------------------------------------

        # Flock full: go home
        if len(self.flock) >= constants.MAX_FLOCK_SIZE:
            self.target_animal = None
            self.turns_chasing_target = 0
            action = Move(*self.move_towards(*self.ark_position))
        else:
            best_animal = self._find_best_animal()
            if best_animal is not None and id(best_animal) not in self.seen_animals:
                if self.turns_chasing_target == self.max_chase_turns:
                    # print(
                    #     f"{self.id} giving up on {best_animal} after {self.max_chase_turns} turns"
                    # )
                    self.seen_animals.add(id(best_animal))
                    self.target_animal = None
                    self.turns_chasing_target = 0
                    action = Move(*self._get_angled_move())
                else:
                    best_animal_pos = self._find_animal_position(best_animal)
                    if best_animal_pos is not None:
                        cell_x, cell_y = int(best_animal_pos[0]), int(best_animal_pos[1])
                        my_cell_x, my_cell_y = int(self.position[0]), int(self.position[1])

                        if cell_x == my_cell_x and cell_y == my_cell_y:
                            self.seen_animals.add(id(best_animal))
                            # print(f"{self.id} obtaining {best_animal}")
                            action = Obtain(best_animal)
                        else:
                            # print(
                            #     f"{self.id} moving toward {best_animal} "
                            #     f"(turn {self.turns_chasing_target}/{self.max_chase_turns})"
                            # )
                            self.turns_chasing_target += 1
                            action = Move(*self.move_towards(*best_animal_pos))
                    else:
                        self.target_animal = None
                        self.turns_chasing_target = 0
                        action = Move(*self._get_angled_move())
            else:
                new_x, new_y = self._get_angled_move()
                # if self.id == 6:
                #     print(f"{self.id} exploring to ({new_x:.2f}, {new_y:.2f})")
                action = Move(new_x, new_y)

        return action
