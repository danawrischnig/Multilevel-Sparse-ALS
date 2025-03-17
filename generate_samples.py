import argparse
import numpy as np
from tqdm import tqdm
from darcy1d import Darcy1D
from datetime import datetime


def qoi(u):
    """Compute the quantity of interest: integral using the trapezoidal rule."""
    return np.trapz(u, dx=1 / (u.shape[-1] - 1), axis=-1)


def generate_data(l, d, points, desc):
    """Solve the 1D Darcy problem for given points and return computed values."""
    problem = Darcy1D(l=l, d=d)
    solutions = [np.copy(problem.solve(y).x.array) for y in tqdm(points, desc=desc)]
    return qoi(np.asarray(solutions))


def main():
    # Generate a timestamp for the default output file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_file = f"data/{timestamp}_samples.npz"

    parser = argparse.ArgumentParser(description="Generate and save Darcy1D data.")
    parser.add_argument(
        "--l_test", type=int, default=14, help="Level of discretization for test data."
    )
    parser.add_argument(
        "--d_test",
        type=int,
        help="Truncation parameter/dimension for test data",
    )
    parser.add_argument(
        "--N_test", type=int, default=1000, help="Number of test points."
    )
    parser.add_argument(
        "--l",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7],
        help="List of discretization levels for training data. Default: [3, 4, 5, 6, 7]",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=6,
        help="Truncation parameter/dimension for training data",
    )
    parser.add_argument(
        "--N", type=int, default=10000, help="Number of training points."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=default_output_file,
        help="Output file name. Default: fem_samples_YYYY-MM-DD_HH-MM-SS.npz",
    )
    args = parser.parse_args()

    args.d_test = args.d_test or args.d
    levels = sorted(args.l) + [args.l_test]

    # Generate test data
    # TODO: When everthing works, test values just needed for test level
    points_test = np.random.uniform(-1, 1, (args.N_test, args.d_test))
    values_test = np.zeros((len(levels), args.N_test))

    for level_idx, l in enumerate(levels):
        values_test[level_idx] = generate_data(
            l,
            args.d_test,
            points_test,
            f"Solve Darcy 1D on mesh of discretization level l = {l} at test points.",
        )

    # Generta training data
    points_train = np.random.uniform(-1, 1, (args.N, args.d))
    values_train = np.zeros((len(levels), args.N))

    for level_idx, l in enumerate(levels):
        values_train[level_idx] = generate_data(
            l,
            args.d,
            points_train,
            f"Solve Darcy 1D on mesh of discretization level l = {l} at training points.",
        )

    # Save data
    np.savez(
        args.output_file,
        levels=levels,
        points_train=points_train,
        points_test=points_test,
        values_train=values_train,
        values_test=values_test,
    )

    print(f"FEM test and training data saved successfully to '{args.output_file}'")
    return args.output_file


if __name__ == "__main__":
    output_filename = main()
    print(
        output_filename
    )  # Print only the filename, so it can be captured by another script
