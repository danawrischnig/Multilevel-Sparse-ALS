import argparse
import numpy as np
from numpy.polynomial.legendre import legval
from collections import defaultdict
import matplotlib.pyplot as plt


def evaluate(points, components):
    """
    Evaluates the function represented by the ALS components at given points.

    Args:
        points (ndarray): Points where the function will be evaluated.
        components (list of ndarray): Tensor train components representing the function.

    Returns:
        ndarray: Evaluated function values at the input points.
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


parser = argparse.ArgumentParser(description="User Info Script")
parser.add_argument(
    "--samples_file",
    type=str,
    default="data/2025-03-15_12-01-39_samples.npz",
    help="File containiong TT for solutions",
)
parser.add_argument(
    "--solution_components_file",
    type=str,
    default="data/2025-03-15_12-01-39_solution_components.npz",
    help="File containiong TT for refinements",
)
parser.add_argument(
    "--refinement_components_file",
    type=str,
    default="data/2025-03-15_12-01-39_refinement_components.npz",
    help="File containiong FEM Samples",
)

args = parser.parse_args()
samples_file = args.samples_file
solution_components_file = args.solution_components_file
refinement_components_file = args.refinement_components_file

# Load FEM sample data
with np.load(samples_file) as sample_data:
    points_test = sample_data["points_test"]
    values_test = sample_data["values_test"][-1]
    d_train = sample_data["points_train"].shape[1]

# Load the data
components_solutions = defaultdict(lambda: defaultdict(lambda: [None] * d_train))

with np.load(solution_components_file) as data:
    for k in data.keys():
        try:
            l, N, mode = map(int, k.split("-"))  # Convert key components to integers
            components_solutions[l][N][mode] = data[k]
        except ValueError as e:
            raise ValueError(
                f"Invalid key format in file: {k}. Expected 'l-N-mode'."
            ) from e

# Convert the defaultdict to a standard dictionary
components_solutions = {
    l: {N: TT for N, TT in components_solutions[l].items()}
    for l in components_solutions
}

# Evaluate sparse TT decompostion at test points
solutions_pred = {
    l: {
        N: evaluate(points_test, components_solutions[l][N])
        for N in components_solutions[l]
    }
    for l in components_solutions
}

# Load the data
components_refinements = defaultdict(lambda: defaultdict(lambda: [None] * d_train))

with np.load(refinement_components_file) as data:
    for k in data.keys():
        try:
            l, N, mode = map(int, k.split("-"))  # Convert key components to integers
            components_refinements[l][N][mode] = data[k]
        except ValueError as e:
            raise ValueError(
                f"Invalid key format in file: {k}. Expected 'l-N-mode'."
            ) from e

# Convert the defaultdict to a standard dictionary
components_refinements = {
    l: {N: TT for N, TT in components_refinements[l].items()}
    for l in components_refinements
}

# Evaluate sparse TT decompostion at test points
refinements_pred = {
    l: {
        N: evaluate(points_test, components_refinements[l][N])
        for N in components_refinements[l]
    }
    for l in components_refinements
}

# Define color maps
red_colors = plt.cm.Reds(np.linspace(0.4, 1, 5))  # Shades of blue
blue_colors = plt.cm.Blues(np.linspace(0.4, 1, 4))  # Shades of pink (red)
green_colors = plt.cm.Greens(np.linspace(0.4, 1, 4))  # Shades of orange

plt.figure(figsize=(8, 6))

# Single level method (shades of blue)
for i, (l, preds) in enumerate(solutions_pred.items()):
    if l >= 7:
        Ns = list(preds.keys())
        w = [2**l * N for N in Ns]
        errors = [np.sqrt(np.mean((v - values_test) ** 2)) for v in preds.values()]
        plt.loglog(w, errors, label=f"SL: l={l}", color=red_colors[i % len(red_colors)])

# Multi-level method (shades of pink for N_max=2560, shades of orange for N_max=640)
for N_max, color_map in zip([2560, 640], [blue_colors, green_colors]):
    for i, l_min in enumerate(range(7, 11)):
        v = np.copy(solutions_pred[l_min][N_max])
        w = [2**l_min * N_max]
        errors = [np.sqrt(np.mean((v - values_test) ** 2))]

        l = l_min + 1
        N = N_max // 2

        while l in refinements_pred.keys() and N in refinements_pred[l].keys():
            try:
                v += refinements_pred[l][N]
                w.append((2**l + 2 ** (l - 1)) * N)
                errors.append(np.sqrt(np.mean((v - values_test) ** 2)))

                l += 1
                N //= 2
            except:
                break

        w = np.cumsum(w)
        plt.loglog(
            w,
            errors,
            label=f"ML: l={l_min},...,{l-1}, N_max = {N_max}",
            color=color_map[i % len(color_map)],
        )

plt.xlabel("Estimated work")
plt.ylabel("SMSE")
plt.title("Test Error")
plt.legend()
plt.show()
