import argparse
import numpy as np
from tqdm import tqdm
from SALS_utils import run_SALS, run_SSALS, evaluate
from darcy1d import Darcy1D

# Reproducibility seeds
DEFAULT_SEED_TEST = 42  # for test set (reference/baseline evaluation)
DEFAULT_SEED_TRAIN = 12345  # for training design (parameter samples)


def parse_args():
    """Parse CLI arguments for experiment configuration."""
    parser = argparse.ArgumentParser(description="Run ML SALS experiments")

    parser.add_argument(
        "-lt",
        "--testlevel",
        type=int,
        default=14,
        help="Discretization level of reference test data (degree=2). Default: 14.",
    )
    parser.add_argument(
        "-dt",
        "--testtruncation",
        type=int,
        help="Truncation dimension for diffusion coefficient of test data. "
        "Defaults to training truncation if not given.",
    )
    parser.add_argument(
        "-Nt",
        "--testsize",
        type=int,
        default=25_000,
        help="Number of test points. Default: 25_000",
    )
    parser.add_argument(
        "-lmax",
        "--maxlevel",
        type=int,
        default=12,
        help="Maximal mesh level for training FEM data (degree=1). Default: 12.",
    )
    parser.add_argument(
        "-d",
        "--truncation",
        type=int,
        default=6,
        help="Truncation dimension of diffusion coefficient for *training* FEM. Default: 6",
    )
    parser.add_argument(
        "-N",
        "--samplesizes",
        type=int,
        nargs="+",
        default=[20 * int(2 ** (k / 2) * (k + 1)) for k in range(7)],
        help=f"Numbers of training points per experiment j. Default: "
        f"{[20 * int(2 ** (k / 2) * (k + 1)) for k in range(7)]}",
    )
    parser.add_argument(
        "--als",
        type=str,
        default="SALS",
        help="ALS variant to use: 'SALS' or 'SSALS'. Default: SALS",
    )

    args = parser.parse_args()

    # If no test truncation is specified, use the training truncation.
    args.testtruncation = args.testtruncation or args.truncation

    # Ensure sample sizes are in ascending order (some logic assumes this).
    args.samplesizes.sort()

    return args


def main():
    """Run the full pipeline: generate data, train surrogates, evaluate errors."""
    args = parse_args()

    # ------------------------
    # Generate reference data
    # ------------------------
    # Reference QoI: degree=2, very fine mesh (l=testlevel)
    desc = f"Generate reference data: mesh width = 2^(-{args.testlevel})"
    problem = Darcy1D(l=args.testlevel, d=args.testtruncation, degree=2)

    # Fixed test set for fair comparisons across methods/levels.
    np.random.seed(DEFAULT_SEED_TEST)
    testpoints = np.random.uniform(-1, 1, (args.testsize, args.testtruncation))
    testvalues = np.zeros(args.testsize)
    for i in tqdm(range(args.testsize), desc=desc):
        testvalues[i] = problem.get_integrated_solution(testpoints[i])

    # ------------------------------------
    # Compute baseline FEM (degree=1) RMSE
    # ------------------------------------
    # Baseline QoI: degree=1 at the finest *training* mesh (l=maxlevel)
    desc = f"Generate baseline FEM: mesh width = 2^(-{args.maxlevel})"
    problem = Darcy1D(l=args.maxlevel, d=args.testtruncation, degree=1)
    femvalues = np.zeros(args.testsize)
    for i in tqdm(range(args.testsize), desc=desc):
        femvalues[i] = problem.get_integrated_solution(testpoints[i])

    # Baseline discretization error: coarse (deg=1, l=maxlevel) vs reference (deg=2, l=testlevel)
    femerror = np.sqrt(np.mean((femvalues - testvalues) ** 2))
    print(f"FEM error (deg=1, l={args.maxlevel} vs ref): {femerror}")

    # -------------------------
    # Allocate multilevel arrays
    # -------------------------
    num_samplesizes = len(args.samplesizes)
    # Maximum multilevel depth (conservative: leave at least 2 levels above 0)
    L_max = min(num_samplesizes, args.maxlevel - 2)
    if L_max <= 0:
        raise ValueError(
            "L_max computed as <= 0. Increase maxlevel or provide more samplesizes."
        )

    # Choose ALS routine
    if args.als == "SALS":
        run_ALS = run_SALS
    elif args.als == "SSALS":
        run_ALS = run_SSALS
    else:
        raise ValueError(f"Unknown ALS variant: {args.als}")

    # Shapes:
    #   predicted0[L, j, :]          : base predictions at level L (coarsest term)
    #   predictedrefinements[L-1, j] : learned refinements between levels (telescoping)
    predicted0 = np.zeros((L_max, num_samplesizes, args.testsize))
    predictedrefinements = np.zeros((L_max - 1, num_samplesizes - 1, args.testsize))

    # Outputs per (L, j):
    predictedvalues = np.zeros((L_max, num_samplesizes, args.testsize))
    work = np.zeros((L_max, num_samplesizes))  # analytical work model
    er = np.zeros((L_max, num_samplesizes))  # RMSE( prediction vs reference )

    # ------------------
    # Build training data
    # ------------------
    # Independent parameter samples per sample size (not nested).
    rng = np.random.default_rng(DEFAULT_SEED_TRAIN)
    trainingpoints = [
        rng.uniform(-1, 1, (N, args.truncation)) for N in args.samplesizes
    ]
    trainingvalues = [np.zeros(N) for N in args.samplesizes]  # will be filled per level

    # -------------------------
    # Single-Level method (L=0)
    # -------------------------
    # Train at finest training mesh (l=maxlevel, degree=1) for each sample size j.
    l = args.maxlevel
    problem = Darcy1D(l=l, d=args.truncation, degree=1)
    desc = f"Generate training data (L=0): mesh width = 2^(-{l})"

    for j in range(num_samplesizes):
        N = args.samplesizes[j]
        for i in tqdm(range(N), desc=desc):
            trainingvalues[j][i] = problem.get_integrated_solution(trainingpoints[j][i])

        # Fit surrogate on (points, values) and predict on the fixed test set.
        tensortrain = run_ALS(trainingpoints[j], trainingvalues[j])
        predicted0[0, j] = evaluate(testpoints, tensortrain)

    # Store single-level predictions and work:
    predictedvalues[0] = predicted0[0]
    # Work model for L=0: cost proportional to N * 2^l (example model; adjust if you measure real timings)
    work[0] = np.array([N * 2**l for N in args.samplesizes])

    # Single-level errors vs. reference:
    er[0] = np.sqrt(np.mean((predictedvalues[0] - testvalues) ** 2, axis=1))
    print("Single-level RMSEs (L=0) across sample sizes:\n", er[0])

    # -------------------------
    # Multi-Level method (L>=1)
    # -------------------------
    # For L = 1..L_max-1:
    #   • Evaluate FEM at coarser mesh l = maxlevel - L
    #   • Build refinements = (values from previous level) - (values at current level)
    #   • Learn surrogate for the base term at L and the refinements for intermediate levels
    #   • Combine via telescoping sum to form predictions at depth L
    for L in range(1, L_max):
        l = args.maxlevel - L
        problem = Darcy1D(l=l, d=args.truncation, degree=1)
        desc = f"Generate training data (L={L}): mesh width = 2^(-{l})"

        # Start refinements from the previous level's values (copy to avoid aliasing).
        trainingrefinements = [np.zeros(N) for N in args.samplesizes]
        trainingrefinements[L - 1 :] = [a.copy() for a in trainingvalues[L - 1 :]]

        # Fresh containers for current level's training values at mesh level l.
        trainingvalues = [np.zeros(N) for N in args.samplesizes]

        for j in range(L - 1, num_samplesizes):
            N = args.samplesizes[j]

            # Evaluate FEM at current level l on the *same* parameter points (per j).
            for i in tqdm(range(N), desc=desc):
                v = problem.get_integrated_solution(trainingpoints[j][i])
                trainingvalues[j][i] = v
                # Refinement = values(previous level) - values(current level)
                trainingrefinements[j][i] -= v

            # Base term at depth L (coarsest in the telescoping sum):
            if j >= L:
                tensortrain = run_ALS(trainingpoints[j], trainingvalues[j])
                predicted0[L, j] = evaluate(testpoints, tensortrain)

            # Refinement terms for intermediate levels in the telescoping sum:
            if j < num_samplesizes - 1:
                tensortrain = run_ALS(trainingpoints[j], trainingrefinements[j])
                predictedrefinements[L - 1, j] = evaluate(testpoints, tensortrain)

        # -----------------
        # Combine (telescoping)
        # -----------------
        expected = num_samplesizes - L  # number of valid (j) entries at depth L
        assert predicted0[L, L:].shape[0] == expected

        # Start with the coarsest/base prediction at depth L
        predictedvalues[L, L:] = predicted0[L, L:]
        # Work model for the base term at level l
        work[L, L:] = np.array([N * 2**l for N in args.samplesizes[L:]])

        # Add refinements from shallower depths to complete the telescoping sum
        for j in range(1, L):
            add = predictedrefinements[L - j, L - j : num_samplesizes - j]
            assert add.shape[0] == expected
            predictedvalues[L, L:] += add
            # Work model for refinements; factor 3*2^(l+j-1) is an example analytical weight
            work[L, L:] += np.array(
                [
                    N * 3 * 2 ** (l + j - 1)
                    for N in args.samplesizes[L - j : num_samplesizes - j]
                ]
            )

        # RMSE vs. fine reference for all j valid at this depth L
        er[L] = np.sqrt(np.mean((predictedvalues[L] - testvalues) ** 2, axis=1))

        print(f"\nLevel {L}:")
        print("RMSEs across sample sizes:\n", er[L])
        print(f"Baseline FEM RMSE (for context): {femerror}")
        print("Work model at this level:\n", work[L])

    # ---------------
    # Persist results
    # ---------------
    filename = f"errorvswork{args.als}level{args.maxlevel}.npz"
    np.savez(filename, er=er, work=work, femerror=femerror)
    print(f"\nSaved results to {filename}")


if __name__ == "__main__":
    main()
