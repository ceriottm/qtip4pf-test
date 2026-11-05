import argparse

from . import get_model


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="metatomic-qtip4pf",
        description="Export the q-TIP4P/F water model as a metatomic model file",
    )
    parser.add_argument(
        "-o", "--output", default="qtip4pf.pt", help="output file (default: %(default)s)"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=9.0,
        help="Lennard-Jones and real-space Ewald cutoff, in Angstrom "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--ewald-accuracy",
        type=float,
        default=1e-6,
        help="target relative accuracy of the Ewald sum (default: %(default)s)",
    )
    args = parser.parse_args(args)

    model = get_model(cutoff=args.cutoff, ewald_accuracy=args.ewald_accuracy)
    model.save(args.output)
    print(f"saved q-TIP4P/F model to {args.output}")


if __name__ == "__main__":
    main()
