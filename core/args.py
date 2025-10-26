import random
import argparse
from dataclasses import dataclass
from typing import cast, Protocol
from time import time

from core.player import Player
from players.group1.player import Player1
from players.group10.player import Player10
from players.group2.player import Player2
from players.group3.player import Player3
from players.group4.player import Player4
from players.group5.player import Player5
from players.group6.player import Player6
from players.group7.player import Player7
from players.group8.player import Player8
from players.group9.player import Player9
from players.random_player import RandomPlayer

import core.constants as c


@dataclass
class Args:
    gui: bool
    seed: int
    player: type[Player]
    num_helpers: int
    animals: list[int]
    time: int
    ark: tuple[int, int]


PLAYERS = {
    "r": RandomPlayer,
    "1": Player1,
    "2": Player2,
    "3": Player3,
    "4": Player4,
    "5": Player5,
    "6": Player6,
    "7": Player7,
    "8": Player8,
    "9": Player9,
    "10": Player10,
}


class ParsedNS(Protocol):
    player: str
    seed: str | None
    gui: bool
    num_helpers: int
    animals: list[str] | None
    T: int
    ark: tuple[str, str] | None


def sanitize_seed(org_seed: None | str) -> int:
    if org_seed is None:
        seed = int(time() * 10_000) % 100_000
        print(f"Generated seed: {seed}")
        return seed

    return int(org_seed)


def sanitize_player(org_player: None | str) -> type[Player]:
    if org_player is None:
        print("Using random player")
        return RandomPlayer

    if org_player not in PLAYERS:
        raise Exception(
            f"player `{org_player}` not valid (expected {' '.join(PLAYERS.keys())})"
        )

    return PLAYERS[org_player]


def sanitize_animals(org_animals: None | list[str]) -> list[int]:
    if org_animals is None:
        raise Exception("Missing animal populations")

    animals = list(map(int, org_animals))
    if any([a < 2 for a in animals]):
        raise Exception("all animals must have populations >= 2")

    return animals


def sanitize_time(org_T: None | int) -> int:
    if org_T is None:
        time = random.randint(c.MIN_T, c.MAX_T)
        print(f"generated T={time}")
        return time

    return org_T


def sanitize_ark(org_ark: None | tuple[str, str]) -> tuple[int, int]:
    if org_ark is None:
        x, y = random.randint(0, c.X - 1), random.randint(0, c.Y - 1)
        print(f"generated ark pos={x, y}")
        return x, y

    return int(org_ark[0]), int(org_ark[1])


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Run the Noah's helpers simulator.")

    parser.add_argument(
        "--player",
        choices=list(PLAYERS.keys()),
        help="Which player to run (1-10 or r for random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_helpers",
        type=int,
        default=10,
        help="Number of helpers",
    )
    parser.add_argument(
        "--animals",
        nargs="+",
        metavar="S",
        help="Number of helpers",
    )
    parser.add_argument(
        "-T",
        type=int,
        help="Time, in turns",
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI")

    parser.add_argument(
        "--ark",
        nargs=2,
        metavar=("X", "Y"),
        help="Ark position",
    )

    args = cast(ParsedNS, parser.parse_args())

    seed = sanitize_seed(args.seed)
    player = sanitize_player(args.player)
    animals = sanitize_animals(args.animals)
    time = sanitize_time(args.T)
    ark = sanitize_ark(args.ark)

    return Args(
        player=player,
        seed=seed,
        gui=args.gui,
        num_helpers=args.num_helpers,
        animals=animals,
        time=time,
        ark=ark,
    )
