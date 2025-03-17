import argparse
import time
import re
import numpy as np
from numpy.polynomial.legendre import legval

from loguru import logger
from colored import fg, attr

from semisparse_als import SemiSparseALS as ALS

import pdb


def run_als(points_train, values_train, stagnation_threshold=3, max_iterations=500):
    """
    Runs the Semi-Sparse Alternating Least Squares (ALS) algorithm.

    Args:
        points_train (ndarray): Training input points.
        values_train (ndarray): Corresponding function values.
        stagnation_threshold (int): Number of iterations without improvement before stopping.
        max_iterations (int): Maximum number of iterations allowed.

    Returns:
        list: Tensor train components representing the function.
    """
    N, d = points_train.shape
    factors = np.sqrt(2 * np.arange(d) + 1)
    measures = legval(
        points_train, np.diag(factors)
    ).T  # Compute Legendre polynomial values

    weights = np.ones(N, dtype=float)  # Initialize weights to 1
    weight_sequence = np.tile(factors, (d, 1))
    sparseALS = ALS(
        measures, values_train, weights, weight_sequence, perform_checks=True
    )

    training_errors = [sparseALS.residual()]
    times = [time.process_time()]

    def print_state(it):
        """Prints the current iteration state."""
        itrStr = f"{it:{len(str(max_iterations))}}"
        training_color = (
            "dark_sea_green_2"
            if training_errors[it] <= np.min(training_errors) + 1e-8
            else "misty_rose_3"
        )
        training_str = f"{fg(training_color)}{training_errors[it]:.2e}{attr('reset')}"
        logger.info(f"[{itrStr}]  Residuals: trn={training_str}")

    for it in range(1, max_iterations + 1):
        try:
            sparseALS.step()
        except StopIteration:
            break

        times.append(time.process_time())
        training_errors.append(sparseALS.residual())
        print_state(it)

        if it - np.argmin(training_errors) - 1 > stagnation_threshold:
            logger.warning(
                "Terminating: training errors stagnated for over 3 iterations"
            )
            break

    return sparseALS.components


def main():
    N_train_defaults = [10 * int(np.ceil(2 ** (k / 2))) for k in range(3, 17)]

    parser = argparse.ArgumentParser(
        description="Run Semi-Sparse ALS algorithm and save resulting tensor train components."
    )
    parser.add_argument("input_file", type=str, help="Path to the FEM data file.")
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file name. Default: <prefix>_refinement_components.npz or <prefix>_solution_components.npz",
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=N_train_defaults,
        help="List of number of samples for SALS experiments.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Run SALS for refinements instead of solutions.",
    )
    args = parser.parse_args()

    # Determine output file name
    output_file = args.output_file
    if output_file is None:
        pattern = re.compile(r"^(?P<prefix>.+)_samples\.npz$")
        match = pattern.match(args.input_file)
        if match:
            prefix = match.group("prefix")
            output_file = (
                f"{prefix}_refinement_components.npz"
                if args.refine
                else f"{prefix}_solution_components.npz"
            )
        else:
            raise ValueError("Could not determine output file name.")

    # Load FEM sample data
    sample_data = np.load(args.input_file)

    points_train = sample_data["points_train"]
    values_train = sample_data["values_train"]
    levels_train = sample_data["levels"][:-1]
    N_train = points_train.shape[0]

    if args.refine:
        if not len(levels_train) > 1:
            raise ValueError("Need at least two levels for refinements.")

        values_train = values_train[1:] - values_train[:-1]
        levels_train = levels_train[1:]

    # Ensure we have enough FEM samples
    if sum(args.N) > N_train:
        raise ValueError(f"{sum(args.N)} FEM samples needed, got {N_train}.")

    component_dict = {}

    for l_idx, l in enumerate(levels_train):
        start_idx, end_idx = 0, 0
        for N in args.N:
            logger.info(
                f"Run SL SALS for FEM mesh discretization level l = {l}, number of samples N = {N}"
            )
            start_idx, end_idx = end_idx, end_idx + N

            components = run_als(
                points_train=points_train[start_idx:end_idx],
                values_train=values_train[l_idx][start_idx:end_idx],
            )

            for mode, component in enumerate(components):
                component_dict[f"{l}-{N}-{mode}"] = component

    # Save tensor train components
    np.savez_compressed(output_file, **component_dict)

    logger.info(f"Tensor train components saved successfully to '{output_file}'")
    return output_file


if __name__ == "__main__":
    output_filename = main()
    print(
        output_filename
    )  # Print only the filename, so it can be captured by another script
