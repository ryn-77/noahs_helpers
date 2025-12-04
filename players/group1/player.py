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

from .search_area import equal_area_angles, random_point_in_segment

W, H = 1000, 1000

# Heuristic tuning constants
OPPOSITE_IN_ARK_MULT = 5.0
OPPOSITE_IN_FLOCK_MULT = 3.0
CHECKED_ANIMAL_RADIUS = 2
CHECKED_ANIMAL_EXPIRY_TURNS = 60
NO_PROGRESS_RETURN_TURNS = 400
REVISIT_COOLDOWN_TURNS = 1000
RAIN_WINDOW_TURNS = 1008
RAIN_EPSILON_TURNS = 50


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class Player1(Player):
    def __init__(
        self,
        id: int,
        ark_x: int,
        ark_y: int,
        kind: Kind,
        num_helpers: int,
        species_populations: dict[str, int],  # logically int keys
    ):
        super().__init__(id, ark_x, ark_y, kind, num_helpers, species_populations)

        # Global state
        self.is_raining = False
        self.escaping_wall: bool = False
        self.ark_view = None
        # Rain timing
        self.rain_start_turn: int | None = None
        self.returning_from_radius = False

        self.hellos_received: list[int] = []
        self.chase_too_long_count: list[int] = []

        # Ark memory: species_id (int) -> set of genders present
        self.ark_memory: dict[int, set[Gender]] = {}
        self.memory_timestamp = 0
        self.last_ark_visit_turn = -1
        self.current_turn = 0

        # Animal tracking
        self.seen_animals: set[int] = set()
        self.known_pairs: set[tuple[int, Gender]] = set()

        # Chase state
        self.target_animal: Animal | None = None
        self.turns_chasing_target = 0
        self.max_chase_turns = 5

        # Exploration state
        self.visited: set[tuple[int, int]] = set()
        self.target_pos: tuple[float, float] | None = (ark_x, ark_y)
        self.base_explore_dir: tuple[float, float] = (0.0, 0.0)

        # Species list/index for messaging — now ints
        self.species_list: list[str] = sorted(self.species_populations.keys())
        self.species_index: dict[str, int] = {
            s: i for i, s in enumerate(self.species_list)
        }

        # "Checked unknown animals" memory: (x, y, species_id:int) -> last_turn_checked
        self.checked_animals: dict[tuple[int, int, int], int] = {}

        # Progress tracking (for "no progress → return to ark")
        self.last_animal_added_turn: int = 0
        self.prev_flock_size: int = 0

        # Revisit cooldown for exploration
        # Revisit cooldown for exploration
        self.last_seen: dict[tuple[int, int], int] = {}

        # Noah doesn't need angular sectors, but we still initialize angles
        self.begin_angle = 0.0
        self.end_angle = 2 * pi

        # Radius control: after first return to ark, don't go beyond a safe distance
        self.has_returned_once: bool = False

        SLOWDOWN = 1.20  # must match the one in get_action rain logic
        self.max_explore_radius: float = (
            RAIN_WINDOW_TURNS - RAIN_EPSILON_TURNS
        ) / SLOWDOWN
        self.resume_explore_radius: float = self.max_explore_radius - 100.0

        if self.id == 0:
            # Noah
            return

        # Partition helpers into equal-area angular segments around the ark
        out = equal_area_angles(ark_x, ark_y, num_helpers - 1)
        if self.id == 1:
            self.begin_angle = 0.0
            self.end_angle = out[0]
        elif self.id == num_helpers - 1:
            self.begin_angle = out[-1]
            self.end_angle = 2 * pi
        else:
            self.begin_angle = out[id - 2]
            self.end_angle = out[id - 1]

    # --- Ark state helpers -------------------------------------------------

    def _get_my_cell(self) -> CellView:
        xcell, ycell = tuple(map(int, self.position))
        if not self.sight.cell_is_in_sight(xcell, ycell):
            raise Exception(f"{self} failed to find own cell")
        return self.sight.get_cellview_at(xcell, ycell)

    def _update_ark_memory(self):
        self.ark_memory = {}
        if self.ark_view is None:
            return
        for animal in self.ark_view.animals:
            species_id, gender = animal.species_id, animal.gender
            species = species_id  # int
            if species not in self.ark_memory:
                self.ark_memory[species] = set()
            self.ark_memory[species].add(gender)

        self.memory_timestamp = self.current_turn
        self.last_ark_visit_turn = self.current_turn

    def _species_has_both_genders(self, species: int) -> bool:
        genders = self.ark_memory.get(species, set())
        return Gender.Male in genders and Gender.Female in genders

    def _has_opposite_gender_in_ark(self, species: int, gender: Gender) -> bool:
        genders = self.ark_memory.get(species, set())
        if gender == Gender.Male:
            return Gender.Female in genders
        if gender == Gender.Female:
            return Gender.Male in genders
        return False

    def _has_opposite_gender_in_flock(self, species: int, gender: Gender) -> bool:
        for other in self.flock:
            if other.species_id != species:
                continue
            if gender == Gender.Male and other.gender == Gender.Female:
                return True
            if gender == Gender.Female and other.gender == Gender.Male:
                return True
        return False

    def _is_animal_needed(self, species: int, gender: Gender) -> bool:
        """
        Pair-aware need check:
        - If ark already has both genders for this species → not needed.
        - For Unknown gender: we want it if the species is incomplete.
        - For known gender: we want it if that gender is missing from ark.
        """
        # If ark already has both genders, extra animals are not needed
        if self._species_has_both_genders(species):
            return False

        genders = self.ark_memory.get(species, set())

        # Unknown gender can still be useful if species is incomplete
        if gender == Gender.Unknown:
            # Species not present at all → we want at least one
            if not genders:
                return True
            # If only one gender present, Unknown might be the opposite
            if len(genders) == 1:
                return True
            # Shouldn't reach here because both genders handled above,
            # but keep safe fallthrough:
            return not self._species_has_both_genders(species)

        # For known gender: skip if it's already present in ark
        if gender in genders:
            print(f"{self.id} does not need {species} ({gender}) – ark already has it")
            return False

        # Otherwise we still need this gender
        return True

    def _already_have_in_flock(self, species: int, gender: Gender) -> bool:
        for animal in self.flock:
            if animal.species_id == species and animal.gender == gender:
                print(f"{self.id} already has {animal} in flock – skipping")
                return True
        return False

    # --- Tiny memory messages ----------------------------------------------

    def _encode_memory_message(self) -> int:
        """Encode ark memory into a 1-byte message."""
        if not self.species_list:
            return 0

        species_idx = self.current_turn % len(self.species_list)
        species = self.species_list[species_idx]  # int

        msg = species_idx & 0x0F  # low 4 bits: species index

        genders = self.ark_memory.get(species, set())  # type: ignore
        if Gender.Male in genders:
            msg |= 0x10
        if Gender.Female in genders:
            msg |= 0x20

        time_bits = (self.memory_timestamp % 4) << 6
        msg |= time_bits

        return msg & 0xFF

    def _decode_memory_message(self, msg: int) -> tuple[int, set[Gender], int]:
        species_idx = msg & 0x0F
        genders: set[Gender] = set()
        if msg & 0x10:
            genders.add(Gender.Male)
        if msg & 0x20:
            genders.add(Gender.Female)
        timestamp_mod4 = (msg >> 6) & 0x03
        return species_idx, genders, timestamp_mod4

    def _update_from_message(self, msg_content: int, msg_timestamp_hint: int):
        """Merge another helper's ark memory into ours if newer."""
        species_idx, genders, msg_time_bits = self._decode_memory_message(msg_content)

        if species_idx >= len(self.species_list):
            return

        species = self.species_list[species_idx]  # int

        inferred_timestamp = (msg_timestamp_hint // 4) * 4 + msg_time_bits
        if msg_timestamp_hint % 4 > msg_time_bits and msg_timestamp_hint >= 4:
            inferred_timestamp += 4

        if inferred_timestamp > self.memory_timestamp:
            if species not in self.ark_memory:
                self.ark_memory[species] = set()  # type: ignore
            self.ark_memory[species].update(genders)  # type: ignore
            self.memory_timestamp = inferred_timestamp

    def _animal_priority(self, animal: Animal, cell_x: float, cell_y: float) -> float:
        species = animal.species_id  # int
        gender = animal.gender

        # Skip species already fully satisfied
        if self._species_has_both_genders(species):
            return 0.0

        # Check if we even need this animal (pair-aware)
        if not self._is_animal_needed(species, gender):
            return 0.0

        # Base weight from rarity
        pop = max(self.species_populations.get(species, 1), 1)  # type: ignore
        base = 1.0 / math.sqrt(pop)  # rare species → larger weight

        weight = base
        if self._has_opposite_gender_in_ark(species, gender):
            weight *= OPPOSITE_IN_ARK_MULT
        elif self._has_opposite_gender_in_flock(species, gender):
            weight *= OPPOSITE_IN_FLOCK_MULT

        dist = distance(*self.position, cell_x, cell_y) + 1e-3
        return weight / dist

    def _was_animal_checked_nearby(
        self, x: int, y: int, species: int, radius: int = CHECKED_ANIMAL_RADIUS
    ) -> bool:
        """Have we recently checked an unknown animal of this species near here?"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                key = (x + dx, y + dy, species)
                t = self.checked_animals.get(key)
                if (
                    t is not None
                    and self.current_turn - t < CHECKED_ANIMAL_EXPIRY_TURNS
                ):
                    return True
        return False

    def _find_best_animal(self) -> Animal | None:
        """Find best animal to pursue using pair-aware rarity scoring."""
        best_animal: Animal | None = None
        best_score = 0.0

        for cellview in self.sight:
            cell_center_x = cellview.x + 0.5
            cell_center_y = cellview.y + 0.5
            helpers = cellview.helpers

            # If other helpers are already in this cell, skip this cell
            if any(helper.id != self.id for helper in helpers):
                # Just skip this cell, don't bail out of the whole search
                continue

            for animal in cellview.animals:
                species = animal.species_id
                gender = animal.gender

                # 1) Don't chase exact animals already in our flock (defensive)
                if animal in self.flock:
                    # print(f"{self.id} already has {animal} in flock – skipping")
                    continue

                # 2) Don't get another of the same species+gender if we already have one
                if self._already_have_in_flock(species, gender):
                    # print(f"{self.id} already has {species}/{gender} in flock – skipping")
                    continue

                # 3) Don't re-chase animals we've already interacted with / given up on
                if id(animal) in self.seen_animals:
                    continue

                # 4) If we don't actually need this animal (based on ark state), skip
                if not self._is_animal_needed(species, gender):
                    # print(f"{self.id} does not need {animal} – skipping")
                    continue

                # 5) Avoid repeatedly visiting useless unknowns
                if gender == Gender.Unknown and self._was_animal_checked_nearby(
                    cellview.x, cellview.y, species
                ):
                    # print(f"{self.id} skipping recently checked unknown {animal}...")
                    continue

                # 6) Compute priority score (rarity, pair completion, distance)
                score = self._animal_priority(animal, cell_center_x, cell_center_y)
                if score <= 0.0:
                    # print(f"{self.id} skipping low-priority {animal}...")
                    continue

                if score > best_score:
                    best_score = score
                    best_animal = animal

        return best_animal

    def _find_animal_position(self, animal: Animal) -> tuple[float, float] | None:
        for cellview in self.sight:
            if animal in cellview.animals:
                return (cellview.x + 0.5, cellview.y + 0.5)
        return None

    def _turns_to_ark(self) -> int:
        return math.ceil(distance(*self.position, *self.ark_position))

    # --- Wall & exploration helpers ----------------------------------------

    def _is_near_wall(self, margin: float = 1.0) -> bool:
        x, y = self.position
        return (
            x <= margin
            or y <= margin
            or x >= float(W) - margin
            or y >= float(H) - margin
        )

    def _is_safely_away_from_wall(self, safe_margin: float = 100.0) -> bool:
        x, y = self.position
        return (
            safe_margin <= x <= float(W) - safe_margin
            and safe_margin <= y <= float(H) - safe_margin
        )

    def _get_away_from_wall_move(self) -> tuple[float, float]:
        target_x, target_y = self.ark_position
        if self.can_move_to(target_x, target_y):
            return target_x, target_y
        return self.position[0] + 1.0, self.position[1] + 1.0

    def _get_random_move(self) -> tuple[float, float]:
        old_x, old_y = self.position
        dx, dy = random() - 0.5, random() - 0.5

        while not self.can_move_to(old_x + dx, old_y + dy):
            dx, dy = random() - 0.5, random() - 0.5

        return old_x + dx, old_y + dy

    def _get_move_direction(self) -> tuple[float, float]:
        """Compute base exploration direction (optional use)."""
        if self.kind == Kind.Noah or self.num_helpers <= 0:
            self.base_explore_dir = (0.0, 0.0)
            return self.base_explore_dir

        ark_x, ark_y = self.ark_position
        map_center_x = float(W) / 2.0
        map_center_y = float(H) / 2.0

        if ark_x == map_center_x and ark_y == map_center_y:
            angle = 2.0 * pi * (self.id / self.num_helpers)
            self.base_explore_dir = (math.cos(angle), math.sin(angle))
            return self.base_explore_dir

        dist_to_left = ark_x
        dist_to_right = float(W) - ark_x
        dist_to_bottom = ark_y
        dist_to_top = float(H) - ark_y

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

        idx = self.id % max(self.num_helpers, 1)

        base_angle = atan2(best_quadrant[1], best_quadrant[0])
        spread = (pi / 1.5) * (idx / max(self.num_helpers - 1, 1)) - (pi / 4.0)
        final_angle = base_angle + spread

        self.base_explore_dir = (math.cos(final_angle), math.sin(final_angle))
        return self.base_explore_dir

    def _recently_seen_cells(self) -> set[tuple[int, int]]:
        cutoff = self.current_turn - REVISIT_COOLDOWN_TURNS
        return {cell for cell, t in self.last_seen.items() if t > cutoff}

    def _meander_in_segment(self) -> tuple[float, float]:
        """
        Sector-constrained exploration:
        - If we have no target or we reached it, pick a new one in our segment
          using _pick_new_target_in_segment (pseudo-random but systematic).
        - Otherwise, keep heading toward the current target.
        """
        current_pos = self.position

        # Need an initial target or we've reached the current one
        if self.target_pos is None or distance(*current_pos, *self.target_pos) < 0.5:
            new_target = self._pick_new_target_in_segment()

            return new_target

        # Keep current target
        return self.target_pos

    def _pick_new_target_in_segment(self) -> tuple[float, float]:
        """
        Pick a new target point in this helper's angular segment
        around the ark, avoiding recently seen cells to make the
        search more systematic over time.
        """
        origin_x, origin_y = self.ark_position  # search fan anchored at ark
        recent = self._recently_seen_cells()
        attempts = 0

        # Initialize candidate so the fallback has a defined value even if the loop
        # exits without assigning (defensive against static analyzers).
        candidate = random_point_in_segment(
            origin_x, origin_y, self.begin_angle, self.end_angle
        )

        while attempts < 200:
            attempts += 1
            candidate = random_point_in_segment(
                origin_x, origin_y, self.begin_angle, self.end_angle
            )
            cx, cy = int(candidate[0]), int(candidate[1])
            if (cx, cy) not in recent:
                self.target_pos = candidate
                return candidate

        # Fallback: if everything in the segment is "recent", just accept the last one
        self.target_pos = candidate
        return candidate

    # --- Progress-based return-to-ark logic --------------------------------

    def _should_return_for_progress(self) -> bool:
        """
        If we've got some animals but haven't added any new animals to our
        flock in a long time, head back to the ark to deposit and refresh
        ark_memory.
        """
        if len(self.flock) == 0:
            return False
        return (
            self.current_turn - self.last_animal_added_turn
        ) >= NO_PROGRESS_RETURN_TURNS

    # --- Main environment hooks --------------------------------------------

    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot) -> int:
        self.position = snapshot.position
        self.flock = snapshot.flock

        # Mark flock animals as "seen" so we don't re-target them
        for a in self.flock:
            self.seen_animals.add(id(a))

        self.sight = snapshot.sight
        prev_raining = self.is_raining
        self.is_raining = snapshot.is_raining
        self.ark_view = snapshot.ark_view

        self.current_turn += 1

        # Record the first turn on which we see rain
        if self.is_raining and not prev_raining and self.rain_start_turn is None:
            self.rain_start_turn = self.current_turn

        # Track progress: did we add animals this turn?
        if len(self.flock) > self.prev_flock_size:
            self.last_animal_added_turn = self.current_turn
        self.prev_flock_size = len(self.flock)

        # Mark animals in our current cell as "checked" (especially Unknown)
        try:
            xcell, ycell = tuple(map(int, self.position))
            cellview = self._get_my_cell()
            for animal in cellview.animals:
                key = (xcell, ycell, animal.species_id)
                self.checked_animals[key] = self.current_turn
        except Exception:
            # If we fail to get cell view, just skip marking
            pass

        # Track which cells we've seen recently
        if self.sight is not None:
            for cellview in self.sight:
                self.visited.add((cellview.x, cellview.y))
                self.last_seen[(cellview.x, cellview.y)] = self.current_turn

        if self.is_in_ark():
            self._update_ark_memory()
            # After visiting the ark, next exploration uses a fresh segment target
            self.target_pos = None
            # Once we've returned at least once, we enforce a safe exploration radius.
            self.has_returned_once = True

        msg = self._encode_memory_message()
        if not self.is_message_valid(msg):
            msg = msg & 0xFF

        return msg

    def get_action(self, messages: list[Message]) -> Action | None:
        # Merge ark memory from other helpers' 1-byte messages
        for msg in messages:
            self._update_from_message(msg.contents, self.current_turn)

        # Reset wall escape once we are well away from the walls
        if self.is_in_ark():
            self.escaping_wall = False

        # Noah doesn't move
        if self.kind == Kind.Noah:
            return None

        # ---------------------- RAIN RETURN LOGIC ----------------------
        if self.is_raining:
            # Record first rainy turn if not done already
            if self.rain_start_turn is None:
                self.rain_start_turn = self.current_turn

            turns_since_rain = self.current_turn - self.rain_start_turn
            remaining_turns = max(0, RAIN_WINDOW_TURNS - turns_since_rain)

            # Euclidean distance (correct metric for this simulator)
            dist = distance(*self.position, *self.ark_position)

            # Slowdown factor ~1.15–1.25 models realistic zig-zag + clipping inefficiency
            SLOWDOWN = 1.20
            turns_to_ark = math.ceil(dist * SLOWDOWN)

            # If we are close to the "latest safe leave" moment, go home now
            if remaining_turns <= turns_to_ark + RAIN_EPSILON_TURNS:
                return Move(*self.move_towards(*self.ark_position))

        # ---------------------- Wall escape & safety ------------------------

        if self.escaping_wall:
            return Move(*self.move_towards(*self.ark_position))

        if self._is_near_wall():
            print(f"{self.id} near wall, initiating escape")
            self.escaping_wall = True
            # After hitting a wall, forget current exploration target so we
            # pick a fresh one in our segment next time we explore.
            self.target_pos = None
            return Move(*self.move_towards(*self.ark_position))

        # Flock full -------------------------------------
        if len(self.flock) >= constants.MAX_FLOCK_SIZE:
            self.target_animal = None
            self.turns_chasing_target = 0
            return Move(*self.move_towards(*self.ark_position))

        # If we've been wandering with animals but no progress, go home
        if self._should_return_for_progress():
            return Move(*self.move_towards(*self.ark_position))

        # ---------------------- SAFE EXPLORATION CONTROL ------------------------
        dist = distance(*self.position, *self.ark_position)

        if self.has_returned_once:
            # After first successful return, never go beyond max_explore_radius
            R_SOFT = self.max_explore_radius
            R_RESUME = self.resume_explore_radius
        else:
            # Before first return we don't constrain by radius here
            R_SOFT = float("inf")
            R_RESUME = float("inf")

        # Determine radius-based return mode
        if dist >= R_SOFT:
            # Enter (or remain in) “forced inward” mode
            self.returning_from_radius = True
        elif dist <= R_RESUME:
            # Safe to turn exploration back on
            self.returning_from_radius = False

        # If we are currently forcing an inward return:
        if self.returning_from_radius:
            # Hard boundary is automatically handled because R_HARD > R_SOFT
            x, y = self.position
            ax, ay = self.ark_position

            dx, dy = ax - x, ay - y
            mag = math.hypot(dx, dy)
            if mag > 0:
                dx /= mag
                dy /= mag

            # Direct inward step
            tx = x + dx
            ty = y + dy

            # If blocked, try rotated inward alternatives
            if not self.can_move_to(tx, ty):
                for k in [0.2, -0.2, 0.4, -0.4]:
                    rx = x + dx + k * dy
                    ry = y + dy - k * dx
                    if self.can_move_to(rx, ry):
                        tx, ty = rx, ry
                        break

            return Move(tx, ty)


        # Otherwise: search for best animal
        best_animal = self._find_best_animal()
        print(f"{self.id} best animal: {best_animal}")
        if best_animal is not None:
            # Reset chase timer when switching targets
            if self.target_animal is not best_animal:
                self.target_animal = best_animal
                self.turns_chasing_target = 0

            best_animal_pos = self._find_animal_position(best_animal)
            if best_animal_pos is not None:
                cell_x, cell_y = int(best_animal_pos[0]), int(best_animal_pos[1])
                my_cell_x, my_cell_y = int(self.position[0]), int(self.position[1])
                # Same cell: attempt to obtain
                if cell_x == my_cell_x and cell_y == my_cell_y:
                    self.seen_animals.add(id(best_animal))
                    print(f"{self.id} obtaining {best_animal}")
                    self.target_animal = None
                    self.turns_chasing_target = 0
                    return Obtain(best_animal)
                else:
                    print(
                        f"{self.id} moving toward {best_animal} "
                        f"(turn {self.turns_chasing_target}/{self.max_chase_turns})"
                    )
                    return Move(*self.move_towards(*best_animal_pos))

            # Lost sight of target
            self.target_animal = None
            self.turns_chasing_target = 0

        # No good animal found → meander within assigned segment
        new_pos = self._meander_in_segment()
        return Move(*self.move_towards(new_pos[0], new_pos[1]))
