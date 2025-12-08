import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from pack import unpack_tensortrain
from multilevel2 import MultilevelSurrogate


def compute_surrogates(
    L: int, main_dir: Union[str, Path]
) -> Tuple[List[MultilevelSurrogate], List[float]]:
    """
    Build multilevel surrogate models for a given number of levels L.

    This function reads tensor-train components for a sequence of sample sizes
    from disk and assembles a list of `MultilevelSurrogate` instances, together
    with corresponding work estimates.

    The directory `main_dir` is expected to contain:
      - `experiment_args.json` with keys:
          * "d": problem dimension
          * "samplesizes": list of sample sizes
          * "max_discr_level": maximum discretization level
      - `base_tensors/` with files `N{N}.npz` (base level TT components)
      - `refinement_tensors/` with files `N{N}.npz` (refinement TT components)

    Parameters
    ----------
    L:
        Number of multilevel correction layers.
    main_dir:
        Path to the experiment directory containing configuration and tensors.

    Returns
    -------
    surrogates:
        List of assembled `MultilevelSurrogate` objects, one for each
        admissible starting index in the samplesizes list.
    work_estimates:
        List of corresponding work estimates (floats) for each surrogate.
    """
    main_dir = Path(main_dir)

    # Load experiment configuration
    with open(main_dir / "experiment_args.json", "r") as f:
        args: Dict[str, Any] = dict(json.load(f))

    base_dir = main_dir / "base_tensors"
    refine_dir = main_dir / "refinement_tensors"

    num_samplesizes = len(args["samplesizes"])

    surrogates: List[MultilevelSurrogate] = []
    work_estimates: List[float] = []

    for start_idx in range(L - 1, num_samplesizes):
        # Create a new multilevel surrogate with L layers
        u = MultilevelSurrogate(d=args["d"], L=L)

        # Base level
        N = args["samplesizes"][start_idx]
        l = args["max_discr_level"] - L + 1

        with np.load(base_dir / f"N{N}.npz") as data:
            ttcomponents_packed = data[f"l{l}"]
        ttcomponents = unpack_tensortrain(ttcomponents_packed)

        u.set_layer(0, ttcomponents)
        work = N * 2**l

        # Refinement levels
        for layer_idx in range(1, L):
            N = args["samplesizes"][start_idx - layer_idx]
            l += 1
            work += N + (2**l + 2 ** (l - 1))

            with np.load(refine_dir / f"N{N}.npz") as data:
                ttcomponents_packed = data[f"l{l}"]
            ttcomponents = unpack_tensortrain(ttcomponents_packed)
            u.set_layer(layer_idx, ttcomponents)

        surrogates.append(u)
        work_estimates.append(work)

    return surrogates, work_estimates


def compute_experiment_data(
    main_dir: Union[str, Path],
    L_max: int,
    testpoints: np.ndarray,
    testvalues: np.ndarray,
    testvalues_coarse: np.ndarray,
) -> Path:
    """
    Compute RMSE vs. work data for all multilevel depths and store in an NPZ file.

    For each multilevel depth L = 1, ..., L_max, this function:
      1. Builds surrogates and work estimates via `compute_surrogates`.
      2. Evaluates surrogates at the given test points.
      3. Computes RMSE against the finest reference test values.
      4. Stores all work/RMSE arrays and FEM reference errors in
         `rmse_work_data.npz` located in `main_dir`.

    Parameters
    ----------
    main_dir:
        Path to the experiment directory (contains tensors and args).
    L_max:
        Maximum number of multilevel correction layers to be considered.
    testpoints:
        Array of shape (n_test, d) with parameter samples.
    testvalues:
        Reference values at `testpoints` for the finest discretization.
    testvalues_coarse:
        Reference values at `testpoints` for a fixed coarse discretization.

    Returns
    -------
    npz_path:
        Path to the written NPZ file containing all work/RMSE data.
    """
    main_dir = Path(main_dir)

    # Load experiment configuration
    with open(main_dir / "experiment_args.json", "r") as f:
        args: Dict[str, Any] = dict(json.load(f))

    print(args)

    # FEM reference error (finest vs. coarse discretization)
    fem_error = np.sqrt(np.mean((testvalues - testvalues_coarse) ** 2))

    # Container for all data to be stored in NPZ
    npz_data: Dict[str, Any] = {"fem_error": fem_error}

    # Loop over multilevel depths
    for L in range(1, L_max + 1):
        surrogates, work_estimates = compute_surrogates(L, main_dir)
        work_estimates_arr = np.asarray(work_estimates)

        # Evaluate surrogates at test points
        preds = np.array([u(testpoints) for u in surrogates])
        rmse = np.sqrt(np.mean((preds - testvalues[None, :]) ** 2, axis=1))

        # Sort by work so lines appear monotone in the plot
        order = np.argsort(work_estimates_arr)
        w_sorted = work_estimates_arr[order]
        rmse_sorted = rmse[order]

        # Store data in NPZ structure
        npz_data[f"work_L{L}"] = w_sorted
        npz_data[f"rmse_L{L}"] = rmse_sorted

        # For L = 1, also store the ALS error against the coarse FEM
        if L == 1:
            als_rmse = np.sqrt(
                np.mean(
                    (preds - testvalues_coarse[None, :]) ** 2,
                    axis=1,
                )
            )
            als_rmse_sorted = als_rmse[order]
            npz_data["als_rmse_L1"] = als_rmse_sorted

    # Save raw work / RMSE data
    npz_outfile = main_dir / "rmse_work_data.npz"
    np.savez(npz_outfile, **npz_data)

    return npz_outfile


def plot_experiment(
    npz_path: Union[str, Path],
    args_path: Union[str, Path],
    outfile: Union[str, Path, None] = None,
) -> None:
    """
    Plot RMSE vs. work curves from a precomputed NPZ file and experiment args.

    This function:
      1. Reads the NPZ file with keys:
           - "fem_error"
           - "als_rmse_L1" (optional)
           - "work_L{L}" and "rmse_L{L}" for multiple L.
      2. Reads the JSON experiment configuration for meta information
         (dimension, discretization level, ALS name, etc.).
      3. Produces a log-log plot of RMSE vs. work and saves it as PNG.

    Parameters
    ----------
    npz_path:
        Path to the NPZ file containing work/RMSE data.
    args_path:
        Path to the `experiment_args.json` file for this experiment.
    outfile:
        Optional path to the output PNG file. If None, a file named
        `rmse.png` is written into the same directory as `npz_path`.
    """
    npz_path = Path(npz_path)
    args_path = Path(args_path)

    # Load configuration
    with open(args_path, "r") as f:
        args: Dict[str, Any] = dict(json.load(f))

    # Load data
    data = np.load(npz_path)

    # Set up figure
    fig = plt.figure()
    als_name = args["als"].upper()
    max_discr_level = args["max_discr_level"]
    d = args["d"]

    weights = r"$w_{\nu} = \prod_j\sqrt{2\nu_j + 1}$"
    if args.get("radius_option") == 2:
        weights = r"$w_{\nu} = \prod_j \sqrt{2\nu_j + 1}\, 1.01\, \nu_j^{3/4}$"

    plt.title(f"{als_name}, dim={d}, finest mesh size: 2^{max_discr_level}, {weights}")

    # FEM reference error
    fem_error = float(data["fem_error"])
    plt.axhline(
        y=fem_error,
        color="gray",
        linestyle="--",
        label="FEM error",
    )

    # Determine all L for which we have data
    L_values = sorted(
        int(key.split("L")[1]) for key in data.files if key.startswith("work_L")
    )

    # Plot curves for each L
    for L in L_values:
        w_sorted = data[f"work_L{L}"]
        rmse_sorted = data[f"rmse_L{L}"]

        # For L = 1, optionally plot the ALS error vs. coarse FEM
        if L == 1 and "als_rmse_L1" in data.files:
            als_rmse_sorted = data["als_rmse_L1"]
            plt.loglog(
                w_sorted,
                als_rmse_sorted,
                "--",
                label=f"{als_name} error",
            )

        plt.loglog(w_sorted, rmse_sorted, "o-", label=f"L = {L}")

    plt.xlabel("work")
    plt.ylabel("RMSE")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()

    # Decide on output file
    if outfile is None:
        outfile = npz_path.parent / "rmse.png"
    else:
        outfile = Path(outfile)

    fig.savefig(outfile, dpi=300, bbox_inches="tight")


def run_full_pipeline() -> None:
    """
    Run the full experiment pipeline for a fixed test set and multiple seeds.

    This function:
      1. Constructs test points and reference values using `Darcy1D`.
      2. Loops over seeds and experiment configurations.
      3. For each combination, explicitly:
         - computes NPZ data,
         - plots using the NPZ and JSON configuration.
    """
    from darcy1d import Darcy1D
    from tqdm import tqdm

    testsize = 1_000
    test_discr_level = 14
    test_discr_degree = 2
    d = 6

    # Sample test points and allocate arrays for reference evaluations
    testpoints = np.random.uniform(-1, 1, (testsize, d))
    testvalues = np.zeros(testsize)
    testvalues_coarse = np.zeros(testsize)

    # Finest and coarse Darcy problems
    testproblem = Darcy1D(test_discr_level, d, test_discr_degree)
    testproblem_coarse = Darcy1D(10, d, 1)

    # Compute reference values
    for i in tqdm(range(testsize)):
        testvalues[i] = testproblem.get_integrated_solution(testpoints[i])
        testvalues_coarse[i] = testproblem_coarse.get_integrated_solution(testpoints[i])

    # Experiment configurations to run for each seed
    experiment_bases = [
        "experiments0/SALS_r1",
        "experiments0/SALS_r2",
        "experiments0/SSALS_r1",
        "experiments0/SSALS_r2",
    ]

    for s in range(9):
        print("=" * 20)
        print("seed", s)
        print("=" * 20)

        for base in experiment_bases:
            exp_dir = Path(base) / f"seed{s}"

            # Step 1: compute and store experiment data
            npz_path = compute_experiment_data(
                exp_dir,
                5,
                testpoints,
                testvalues,
                testvalues_coarse,
            )

            # Step 2: plot from NPZ + JSON
            args_path = exp_dir / "experiment_args.json"
            plot_experiment(npz_path, args_path)


def main() -> None:
    """
    Command-line entry point.

    Usage
    -----
    1) Default (backwards compatible, full pipeline):

        python script.py

    2) Plot only from existing experiments (no recomputation):

        python script.py --plot-only EXP_DIR [EXP_DIR ...]

       where each EXP_DIR contains
         - rmse_work_data.npz
         - experiment_args.json
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Multilevel surrogate experiment driver."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Only plot from existing rmse_work_data.npz and experiment_args.json. "
            "No recomputation of RMSE / work data."
        ),
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="*",
        help=(
            "Experiment directories (e.g. experiments0/SALS_r1/seed0). "
            "Required when using --plot-only."
        ),
    )

    args = parser.parse_args()

    if args.plot_only:
        if not args.experiment_dirs:
            parser.error(
                "Please provide at least one experiment directory when using --plot-only."
            )

        for exp_dir in args.experiment_dirs:
            exp_dir = Path(exp_dir)
            npz_path = exp_dir / "rmse_work_data.npz"
            args_path = exp_dir / "experiment_args.json"

            if not npz_path.is_file():
                raise FileNotFoundError(f"NPZ file not found: {npz_path}")
            if not args_path.is_file():
                raise FileNotFoundError(f"Args JSON file not found: {args_path}")

            print(f"Plotting experiment in {exp_dir} ...")
            plot_experiment(npz_path, args_path)
    else:
        # No flags: run the original full pipeline
        # (ignore any positional arguments to remain simple/backwards compatible)
        if args.experiment_dirs:
            print(
                "Positional arguments are ignored when --plot-only is not set. "
                "Running full pipeline."
            )
        run_full_pipeline()


if __name__ == "__main__":
    main()

# python compute_errors_and_plot.py --plot-only experiments0/SALS_r1/seed0
# python compute_errors_and_plot.py --plot-only experiments0/SALS_r1/seed0 experiments0/SALS_r2/seed0
