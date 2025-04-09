import argparse
import numpy as np
from numpy.polynomial.legendre import legval
from tqdm import tqdm
from darcy1d import Darcy1D
from datetime import datetime
from pathlib import Path
from semisparse_als import SemiSparseALS as ALS


quantities_of_interest = {
    "integral": lambda u: np.trapz(u, dx=1 / (len(u) - 1)),
    "middle value": lambda u: u[len(u) // 2],
}


def parse_args():
    """
    Parse command-line arguments for the ML SALS experiment.

    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="Run ML SALS experiments")

    parser.add_argument(
        "-l_test",
        "--test_level",
        type=int,
        default=14,
        help="Level of FEM mesh discretization for test data. Default: 14.",
    )

    parser.add_argument(
        "-d_test",
        "--test_truncation",
        type=int,
        help="Truncation parameter/dimension of the diffusion coefficient for the computation of the test data. By default same as for the training data.",
    )
    parser.add_argument(
        "-N_test",
        "--test_size",
        type=int,
        default=100_000,
        help="Number of test points. Default: 100_000",
    )

    parser.add_argument(
        "-l_min",
        "--min_training_level",
        type=int,
        default=5,
        help="Minimal level of FEM mesh discretization for training data. Default: 5.",
    )
    parser.add_argument(
        "-l_max",
        "--max_training_level",
        type=int,
        default=12,
        help="Maximal level of FEM mesh discretization for training data. Default: 12.",
    )
    parser.add_argument(
        "-d",
        "--training_truncation",
        type=int,
        default=6,
        help="Truncation parameter/dimension of the diffusion coefficient for the computation of the training data. Deafult: 6",
    )
    parser.add_argument(
        "-N",
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[20 * 2**k for k in range(8)],
        help=f"Numbers of training points, default: {[20 * 2**k for k in range(8)]}",
    )
    parser.add_argument(
        "-qoi",
        "--qoi_key",
        type=str,
        default="integral",
        choices=list(quantities_of_interest.keys()),
        help="Key of quantity of interest. Default: 'integral'.",
    )

    parser.add_argument(
        "-out",
        "--output_path",
        type=str,
        default="experiment_data",
        help="Reltive path to the output directory. Default: 'integral'.",
    )

    args = parser.parse_args()

    # If no test truncation parameter is given, it is set to training truncation parameter
    args.test_truncation = args.test_truncation or args.training_truncation

    # Make sure that sample sizes are sorted
    args.sample_sizes.sort()

    args.training_levels = list(
        range(args.min_training_level, args.max_training_level + 1)
    )
    args.levels = args.training_levels + [args.test_level]
    args.refinement_levels = args.training_levels[1:]

    args.training_size = sum(args.sample_sizes)  # Number of training points

    args.qoi = quantities_of_interest[args.qoi_key]  # Quantity of interest functio

    return args


def generate_samples(
    levels,
    test_size,
    test_truncation,
    training_size,
    training_truncation,
    qoi,
):
    """
    Generate training and test samples, compute FEM solutions, and evaluate the quantity of interest (QoI).

    This function:
    - Generates uniformly random input samples for both test and training sets.
    - Solves the 1D Darcy problem on each mesh level for the test and training data.
    - Evaluates the quantity of interest (QoI) on each solution.

    Args:
        levels (list of int): List of FEM discretization levels (training uses all except the finest).
        test_size (int): Number of test input samples.
        test_truncation (int): Number of random input dimensions for test problems.
        training_size (int): Number of training input samples.
        training_truncation (int): Number of random input dimensions for training problems.
        qoi (Callable): Function that extracts a quantity of interest from a FEM solution array.

    Returns:
        tuple:
            - test_points (ndarray): Random test inputs, shape (test_size, test_truncation).
            - test_values (ndarray): QoI evaluations at test points, shape (len(levels), test_size).
            - training_points (ndarray): Random training inputs, shape (training_size, training_truncation).
            - training_values (ndarray): QoI evaluations at training points, shape (len(levels) - 1, training_size).
    """
    # Prepare numpy arrays
    test_points = np.random.uniform(-1, 1, (test_size, test_truncation))
    training_points = np.random.uniform(-1, 1, (training_size, training_truncation))

    test_values = np.zeros((len(levels), test_size))
    training_values = np.zeros((len(levels) - 1, training_size))

    # Compute FEM solutions on test points
    for l_idx, l in enumerate(levels):
        problem = Darcy1D(l=l, d=test_truncation)
        desc = f"Solve Darcy 1D on mesh of discretization level l = {l} at test points."
        solutions = np.array(
            [np.copy(problem.solve(y).x.array) for y in tqdm(test_points, desc=desc)]
        )

        test_values[l_idx] = np.apply_along_axis(qoi, -1, solutions)

    # Compute FEM solutions on training points
    for l_idx, l in enumerate(levels[:-1]):
        problem = Darcy1D(l=l, d=training_truncation)
        desc = f"Solve Darcy 1D on mesh of discretization level l = {l} at training points."
        solutions = np.array(
            [
                np.copy(problem.solve(y).x.array)
                for y in tqdm(training_points, desc=desc)
            ]
        )

        training_values[l_idx] = np.apply_along_axis(qoi, -1, solutions)

    return test_points, test_values, training_points, training_values


def layer(base, refinements):
    """
    Constructs a layered array by successively adding refinement values to slices
    of the base array.

    Parameters
    ----------
    base : array_like
        A 2D array (N x M) representing the initial base layer.
    refinements : array_like
        A 2D array of shape (N-1, M-1), representing refinement values applied to
        successive layers.

    Returns
    -------
    result : ndarray
        A 3D array of shape (L_max, N, M), where L_max = min(N, M). The first layer
        (index 0) is the base array, and each subsequent layer (index L) is a
        refinement of the previous one by adding slices of `refinements`.

    Notes
    -----
    - The function uses broadcasting and slicing to apply refinements to progressively
      smaller subarrays of the result.
    - Unused entries in the result array remain NaN for clarity.
    """
    base = np.asarray(base)
    refinements = np.asarray(refinements)

    N, M = base.shape[:2]
    assert refinements.shape[:2] == (N - 1, M - 1), "B"
    L_max = min(N, M)
    result_shape = (L_max, N, M)

    if base.ndim > 2:
        num = base.shape[2]
        assert refinements.shape[2] == num, "B"
        result_shape = (L_max, N, M, num)

    result = np.full(result_shape, np.nan)
    result = np.full(result_shape, 0.0)
    result[0, :, :] = base

    for L in range(1, L_max):
        refinement_slice = refinements[L - 1 :, : M - L]
        result[L, : N - L, L:] = result[L - 1, : N - L, L:] + refinement_slice

    return result


def predict(
    training_levels,
    sample_sizes,
    training_points,
    training_values,
    test_points,
    max_it=100,
    stagnation_threshold=3,
):
    """
    Run Semi-Sparse ALS on different mesh levels and sample sizes, and predict test outputs.

    Args:
        training_levels (list of int): FEM discretization levels used for training.
        sample_sizes (list of int): Numbers of training points per level.
        training_points (ndarray): Training input samples.
        training_values (ndarray): Function values at training points.
        test_points (ndarray): Test input samples.
        max_it (int): Maximum number of SALS iterations.
        stagnation_threshold (int): Stopping criterion for ALS stagnation.

    Returns:
        ndarray: Predicted values of shape (num_levels, num_sample_sizes, num_test_points).
    """
    # Prepare prediction array
    test_size = len(test_points)
    num_sample_sizes = len(sample_sizes)
    num_training_levels = len(training_levels)

    predicted_values = np.zeros((num_training_levels, num_sample_sizes, test_size))

    start_idx, end_idx = 0, 0
    for N_idx, N in enumerate(sample_sizes):
        start_idx, end_idx = end_idx, end_idx + N

        desc = f"SALS with {N} samples each of the mesh discretization levels {training_levels[0]} to {training_levels[-1]}."
        for l_idx in tqdm(range(num_training_levels), desc=desc):
            # Define training set and run SALS with it
            points = training_points[start_idx:end_idx]
            values = training_values[l_idx, start_idx:end_idx]
            components = _run_SALS(points, values, stagnation_threshold, max_it)

            # Predict values at test points
            predicted_values[l_idx, N_idx] = _evaluate(test_points, components)

    return predicted_values


def _run_SALS(points, values, stagnation_threshold=3, max_it=500):
    """
    Runs the Semi-Sparse Alternating Least Squares (ALS) algorithm to build a surrogate model.

    Args:
        points (ndarray): Input samples of shape (N, d).
        values (ndarray): Output values of shape (N,).
        stagnation_threshold (int): Number of non-improving iterations before termination.
        max_it (int): Maximum number of ALS iterations.

    Returns:
        list: Tensor-train components representing the learned surrogate model.
    """
    N, d = points.shape
    factors = np.sqrt(2 * np.arange(d) + 1)
    measures = legval(points, np.diag(factors)).T  # Compute Legendre polynomial values

    weights = np.ones(N, dtype=float)  # Initialize weights to 1
    weight_sequence = np.tile(factors, (d, 1))
    sals = ALS(measures, values, weights, weight_sequence, perform_checks=True)

    training_errors = [sals.residual()]

    for it in range(1, max_it + 1):
        try:
            sals.step()
        except StopIteration:
            break

        training_errors.append(sals.residual())

        if it - np.argmin(training_errors) - 1 > stagnation_threshold:
            print("Terminating: training errors stagnated for over 3 iterations")
            break

    return sals.components


def _evaluate(points, components):
    """
    Evaluate a tensor-train surrogate model at given input points using Legendre polynomials.

    Args:
        points (ndarray): Input points of shape (N, d), where N is the number of samples and d is the dimensionality.
        components (list of ndarray): Tensor-train components representing the surrogate model.

    Returns:
        ndarray: Surrogate model evaluations of shape (N,).
    """
    d = len(components)
    leg_sups = np.sqrt(2 * np.arange(components[0].shape[1]) + 1)
    evaluated_legendre = legval(points, np.diag(leg_sups)).T

    result = np.einsum("mi,hij->mj", evaluated_legendre[0], components[0])

    for mode in range(1, d - 1):
        result = np.einsum(
            "mh,mi,hij->mj", result, evaluated_legendre[mode], components[mode]
        )

    return np.einsum("mh,mi,hij->m", result, evaluated_legendre[-1], components[-1])


def compute_MLM_values(predicted_values, predicted_refinements):
    """
    Construct Multi-Level Method (MLM) approximations using base predictions and refinements.

    Args:
        predicted_values (ndarray): Base level predictions of shape (L, S, N).
        predicted_refinements (ndarray): Level-to-level refinements of shape (L-1, S-1, N).

    Returns:
        ndarray: MLM approximations of shape (L_max, L, S, N).
    """
    num_levels, num_sample_sizes, test_size = predicted_values.shape
    L_max = min(num_levels, num_sample_sizes)
    values_MLM = np.zeros((L_max, num_levels, num_sample_sizes, test_size)) * np.nan

    values_MLM[0] = np.copy(predicted_values)

    for L in range(1, L_max):
        lower_level_sum = values_MLM[L - 1, :-L, L:]
        refinements = predicted_refinements[L - 1 :, : num_sample_sizes - L]
        values_MLM[L, :-L, L:] = lower_level_sum + refinements

    return values_MLM


def main():
    """
    Entry point for running the full experiment:
    - Parses CLI arguments
    - Solves the FEM problem
    - Runs the SALS model
    - Computes multi-level surrogates and errors
    - Saves the output to disk
    """
    # Parses CLI arguments
    args = parse_args()

    # Generate and save FEM solutions

    test_points, test_values, training_points, training_values = generate_samples(
        args.levels,
        args.test_size,
        args.test_truncation,
        args.training_size,
        args.training_truncation,
        args.qoi,
    )

    # Run SALS on quantity of interest of the solutions (Single Level Method)
    predicted_values_single_level = predict(
        args.training_levels,
        args.sample_sizes,
        training_points,
        training_values,
        test_points,
    )

    # Run SALS on quantity of interest of the refinements
    # Compute refinements (note that test discretization level excluded)
    # test_refinements = test_values[1:-1] - test_values[:-2]
    training_refinements = training_values[1:, :-1] - training_values[:-1, :-1]
    predicted_refinements = predict(
        args.refinement_levels,
        args.sample_sizes[:-1],
        training_points,
        training_refinements,
        test_points,
    )

    # Construct multi level solutions and compute errors
    predicted_values = layer(predicted_values_single_level, predicted_refinements)

    # Compute errors
    mse = lambda v: np.mean((test_values[-1] - v) ** 2)
    total_errors = np.apply_along_axis(mse, -1, predicted_values)
    fem_errors = np.apply_along_axis(mse, -1, test_values[:-1])

    # Estimate work
    work_single_levels = np.array(
        [[N * 2**l for N in args.sample_sizes] for l in args.training_levels]
    )
    work_refinements = work_single_levels[1:, :-1] + work_single_levels[:-1, :-1]
    work = layer(work_single_levels, work_refinements)

    # Prepare output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"

    # Save predictions
    np.savez(
        output_path / npz_file,
        test_level=args.test_level,
        training_levels=args.training_levels,
        sample_sizes=args.sample_sizes,
        test_points=test_points,
        test_values=test_values,
        training_points=training_points,
        training_values=training_values,
        predicted_values=predicted_values,
        total_errors=total_errors,
        fem_errors=fem_errors,
        work=work,
    )


if __name__ == "__main__":
    main()
