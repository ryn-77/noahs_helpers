import random

from core.runner import ArkRunner
from core.args import parse_args


def main():
    args = parse_args()
    random.seed(args.seed)

    runner = ArkRunner(args.player, args.num_helpers, args.animals, args.time, args.ark)

    if args.gui:
        result = runner.run_gui()
    else:
        result = runner.run()

    print(f"result: {result}")


if __name__ == "__main__":
    main()
