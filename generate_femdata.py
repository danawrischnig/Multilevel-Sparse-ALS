import argparse
import json
from pathlib import Path
import numpy as np
from darcy1d import Darcy1D
from tqdm import tqdm

# Default values
DEFAULT_L = 10


def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment parameters for a Darcy flow PDE setup."
    )

    parser.add_argument(
        "directory",
        type=str,
        help="Directory to save computed tensortrains.",
    )

    parser.add_argument(
        "--l",
        type=int,
        default=DEFAULT_L,
        help="FEM discretization level; the interval is divided into 2**level cells.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main_dir = Path(args.directory)

    with open(main_dir / "experiment_args.json", "r") as f:
        reloaded_args = json.load(f)

    if not 0 <= args.l <= reloaded_args["max_discr_level"]:
        raise ValueError(
            f"FEM level l must be between 0 and {reloaded_args['max_discr_level']}, but got l={args.l}."
        )
    args.d = reloaded_args["d"]
    args.samplesizes = reloaded_args["samplesizes"]

    print("PDE parameters:")
    print(f"  Directory: {args.directory}")
    print(f"  Dimension d: {args.d}")
    print(f"  FEM discretization level l: {args.l}")
    print(f"  Sample sizes: {args.samplesizes}")

    problem = Darcy1D(l=args.l, d=args.d, degree=1)
    desc = f"Compute training values for FEM level {args.l}, dim {args.d}"

    for N in args.samplesizes:
        trainingdata_path = main_dir / "trainingdata" / f"N{N}.npz"
        key = f"l{args.l}"

        # load training points
        with np.load(trainingdata_path) as data:
            if "y" not in data:
                raise KeyError(
                    f"'y' not in {trainingdata_path}, keys={list(data.files)}"
                )
            arrays = dict(data)

        # wenn die Werte für dieses l schon existieren, nichts mehr tun
        if key in arrays:
            print(
                f"training set of size {N} already computed on mesh of size 2^{args.l}"
            )
            continue

        points = arrays["y"]

        # compute training values
        values = np.zeros(N)
        for n in tqdm(range(N), desc=desc):
            values[n] = problem.get_integrated_solution(points[n])

        # save training values
        arrays[key] = values
        np.savez_compressed(trainingdata_path, **arrays)
