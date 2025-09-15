import numpy as np
from numpy.polynomial.legendre import legval
from sparse_als import SparseALS as SALS
from semisparse_als import SemiSparseALS as SSALS


def _run_ALS(points, values, ALS=SALS, stagnation_threshold=3, max_it=500):
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
    measures = legval(points, np.diag(factors)).T  # Compute Legendre polynomial valuess

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


def run_SALS(points, values, stagnation_threshold=3, max_it=500):
    """
    Runs the Sparse Alternating Least Squares (ALS) algorithm to build a surrogate model.

    Args:
        points (ndarray): Input samples of shape (N, d).
        values (ndarray): Output values of shape (N,).
        stagnation_threshold (int): Number of non-improving iterations before termination.
        max_it (int): Maximum number of ALS iterations.

    Returns:
        list: Tensor-train components representing the learned surrogate model.
    """
    return _run_ALS(
        points,
        values,
        ALS=SALS,
        stagnation_threshold=stagnation_threshold,
        max_it=max_it,
    )


def run_SSALS(points, values, stagnation_threshold=3, max_it=500):
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
    return _run_ALS(
        points,
        values,
        ALS=SSALS,
        stagnation_threshold=stagnation_threshold,
        max_it=max_it,
    )


def evaluate(points, components):
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
