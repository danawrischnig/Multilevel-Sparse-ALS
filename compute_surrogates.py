import argparse
import json
from pathlib import Path
import numpy as np
from SALS_utils import run_SALS, run_SSALS
from pack import pack_tensortrain

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
        "--refine",
        action="store_true",
        help="If set, Tensor Train will approximate refinement.",
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
    vars(args).update(reloaded_args)

    print(args)

    if args.radius_option == 1:
        rho = np.ones(args.d)
    else:
        p = 0.75
        rho = 1.01 * np.arange(1, args.d + 1) ** p

    run_ALS = run_SALS if args.als == "SALS" else run_SSALS

    trainingdata_dir = main_dir / "trainingdata"
    if args.refine:
        tensor_dir = main_dir / "refinement_tensors"
        slc = slice(args.max_discr_level - args.l, len(args.samplesizes) - 1)
        if not 1 <= args.l <= args.max_discr_level:
            raise ValueError(
                f"FEM level l must be between 0 and {args.max_discr_level}, but got l={args.l}."
            )
    else:
        tensor_dir = main_dir / "base_tensors"
        slc = slice(args.max_discr_level - args.l, len(args.samplesizes))
        if not 0 <= args.l <= args.max_discr_level:
            raise ValueError(
                f"FEM level l must be between 0 and {args.max_discr_level}, but got l={args.l}."
            )

    for N in args.samplesizes[slc]:
        print("=" * 30)
        print(f"N={N}")
        fn = f"N{N}.npz"
        key = f"l{args.l}"

        if args.refine:
            key_coarse = f"l{args.l - 1}"

        with np.load(tensor_dir / fn) as data:
            if f"l{args.l}" in data:
                print(
                    f"tensor train already computed on traning set of size {N} already computed on mesh of size 2^{args.l}"
                )
                continue

        with np.load(trainingdata_dir / fn) as data:
            if "y" not in data:
                raise KeyError(
                    f"'y' not in {trainingdata_dir / fn}, keys={list(data.files)}"
                )
            if key not in data:
                raise KeyError(
                    f"{key} not in {trainingdata_dir / fn}, keys={list(data.files)}"
                )

            trainingpoints = data["y"]
            trainingvalues = data[key]

            if args.refine:
                if key_coarse not in data:
                    raise KeyError(
                        f"{key_coarse} not in {trainingdata_dir / fn}, keys={list(data.files)}"
                    )
                trainingvalues -= data[key_coarse]

        ttcomponents = run_ALS(trainingpoints, trainingvalues, rho)

        # save tensor train
        with np.load(tensor_dir / fn) as data:
            arrays = dict(data)

        arrays[key] = pack_tensortrain(ttcomponents)
        np.savez_compressed(tensor_dir / fn, **arrays)


"""
    for idx in range(len(args.samplesizes) - args.L + 1):
        # u = MultilevelSurrogate(d=args.d, L=args.L)

        N_idx = idx + args.L
        discr_level = args.max_discr_level - args.L
        work = 0
        for l in range(args.L):
            discr_level += 1
            N_idx -= 1
            N = args.samplesizes[N_idx]

            ttcomponents = []
            # TODO
"""
