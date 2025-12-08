import argparse
from pathlib import Path
import json
import numpy as np


# Default values
DEFAULT_D = 6
DEFAULT_MAX_DISCRETIZATION_LEVEL = 10
DEFAULT_SAMPLESIZES = [10 * int(2 ** (k / 2)) for k in range(4, 15)]
DEFAULT_RADIUS_OPTION = 1  # valid: 1 and 2
DEFAULT_ALS = "SALS"  # valid: "SALS", "SSALS"
DEFAULT_SEED = 42


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
        "--d",
        type=int,
        default=DEFAULT_D,
        help="Dimension/truncation of the diffusion coefficient in the Darcy flow PDE.",
    )
    parser.add_argument(
        "--max_discr_level",
        type=int,
        default=DEFAULT_MAX_DISCRETIZATION_LEVEL,
        help="Maximum FEM discretization level; the interval is divided into 2**level cells.",
    )
    parser.add_argument(
        "--samplesizes",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLESIZES,
        help=f"Space-separated training sample sizes (default: {DEFAULT_SAMPLESIZES}).",
    )
    parser.add_argument(
        "--radius_option",
        type=int,
        default=DEFAULT_RADIUS_OPTION,
        help=f"Radius option for SALS/SSALS (default: {DEFAULT_RADIUS_OPTION}).",
    )
    parser.add_argument(
        "--als",
        type=str,
        default=DEFAULT_ALS,
        help=f"ALS variant to use: 'SALS' or 'SSALS' (default: '{DEFAULT_ALS}').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("Experiment parameters:")
    print(f"  Directory: {args.directory}")
    print(f"  Dimension d: {args.d}")
    print(f"  Max discretization level: {args.max_discr_level}")
    print(f"  Sample sizes: {args.samplesizes}")
    print(f"  Radius option: {args.radius_option}")
    print(f"  ALS variant: {args.als}")
    print(f"  Random seed: {args.seed}")

    main_dir = Path(args.directory)
    main_dir.mkdir(parents=True, exist_ok=True)

    # save args to a file in main directory for future reference (json)
    with open(main_dir / "experiment_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    print("Experiment arguments saved in experiment_args.json")

    trainingdata_dir = main_dir / "trainingdata"
    trainingdata_dir.mkdir(parents=True, exist_ok=False)

    base_dir = main_dir / "base_tensors"
    base_dir.mkdir(parents=True, exist_ok=False)

    refinement_dir = main_dir / "refinement_tensors"
    refinement_dir.mkdir(parents=True, exist_ok=False)

    # generate and save training data points
    np.random.seed(args.seed)
    for N in args.samplesizes:
        fn = f"N{N}.npz"

        data = {"y": np.random.uniform(-1, 1, (N, args.d))}
        np.savez_compressed(trainingdata_dir / fn, **data)

        data = {}
        np.savez_compressed(base_dir / fn, **data)

        data = {}
        np.savez_compressed(refinement_dir / fn, **data)

    print(f"Training data saved in {trainingdata_dir}")
