import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from SALS_utils import run_SALS, run_SSALS, evaluate
from darcy1d import Darcy1D
from pack import load_tensortrain, save_tensortrain
import matplotlib.pyplot as plt
from multilevel import MultilevelSurrogate

# Default values
DEFAULT_LEVEL = 10
DEFAULT_D = 6
DEFAULT_TESTSIZE = 10000
DEFAULT_SAMPLESIZES = [10 * int(2 ** (k / 2)) for k in range(4, 15)]
DEFAULT_RHO_OPTION = 1
DEFAULT_ALS = "SALS"  # valid: "SALS", "SSALS"
DEFAULT_NUMBER_OF_LEVELS = 1


def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment parameters for a Darcy flow PDE setup."
    )

    parser.add_argument(
        "--directory",
        type=str,
        default="sals_experiments",
        help="Directory to save computed tensortrains.",
    )

    parser.add_argument(
        "--level",
        type=int,
        default=DEFAULT_LEVEL,
        help="FEM discretization level; the interval is divided into 2**level cells.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=DEFAULT_D,
        help="Dimension/truncation of the diffusion coefficient in the Darcy flow PDE.",
    )
    parser.add_argument(
        "--testsize",
        type=int,
        default=DEFAULT_TESTSIZE,
        help="Number of test samples to generate/evaluate.",
    )
    parser.add_argument(
        "--samplesizes",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLESIZES,
        help=f"Space-separated training sample sizes (default: {DEFAULT_SAMPLESIZES}).",
    )
    parser.add_argument(
        "--rho_option",
        type=int,
        default=DEFAULT_RHO_OPTION,
        help="Selector for the radius sequence ρ to ensure δ-admissibility.",
    )

    parser.add_argument(
        "--als",
        choices=["SALS", "SSALS"],
        default=DEFAULT_ALS,
        help='ALS variant to use. Valid options: "SALS" or "SSALS".',
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        default=DEFAULT_NUMBER_OF_LEVELS,
        help="Number of discretization levels to use in the multilevel surrogate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("level       :", args.level, "(cells =", 2**args.level, ")")
    print("d           :", args.d)
    print("testsize    :", args.testsize)
    print("samplesizes :", args.samplesizes)
    print("rho_option  :", args.rho_option)
    print("als         :", args.als)

    testpoints = {}

    tp_path = Path(args.directory) / "trainingpoints.npz"
    tp_path.parent.mkdir(parents=True, exist_ok=True)

    # Wenn Datei existiert: vorhandene Punkte laden
    if tp_path.exists():
        # np.load liefert ein NpzFile-Objekt, das sich wie ein dict verhält
        existing = dict(np.load(tp_path))
    else:
        existing = {}

    # Für jede gewünschte Sample-Size ggf. neue Punkte erzeugen
    for N in args.samplesizes:
        key = f"N{N}"
        if key in existing:
            # schon vorhanden → übernehmen
            testpoints[key] = existing[key]
        else:
            # fehlt → neu generieren
            points = np.random.uniform(-1, 1, (N, args.d))
            testpoints[key] = points

    # Alles (alte + neue) wieder speichern
    np.savez_compressed(tp_path, **testpoints)

    # Darcy flow pde solver
    desc = f"Generate reference data: mesh width = 2^(-14), dim = {args.d}, degree = 2"
    problem_fine = Darcy1D(l=14, d=args.d, degree=2)

    # Darcy flow pde solver
    desc = f"Generate reference data: mesh width = 2^(-{args.level}), dim = {args.d}"
    problem = Darcy1D(l=args.level, d=args.d, degree=1)

    if args.rho_option == 2:
        rho = problem.rho

    # Fixed test set for comparisons across sample sizes.
    testpoints = np.random.uniform(-1, 1, (args.testsize, args.d))
    testvalues = np.zeros(args.testsize)
    testvalues_fine = np.zeros(args.testsize)
    for i in tqdm(range(args.testsize), desc=desc):
        testvalues[i] = problem.get_integrated_solution(testpoints[i])
        testvalues_fine[i] = problem_fine.get_integrated_solution(testpoints[i])

    fem_error = np.sqrt(np.mean((testvalues - testvalues_fine) ** 2))

    predictions = []
    work = []
    discrlevels = list(range(args.level - args.num_levels + 1, args.level + 1))

    for i in range(len(args.samplesizes)):
        N = args.samplesizes[i]
        # sub_samplesizes = args.samplesizes[i : i + args.num_levels][::-1]

        sub_samplesizes = [N]
        discrlevels = [args.level]
        surrogate = MultilevelSurrogate(
            discrlevels,
            sub_samplesizes,
            args.d,
            als_variant=args.als,
            rho_option=args.rho_option,
        )
        surrogate.set_directory(args.directory)
        pred = surrogate(testpoints)
        predictions.append(pred)
        work.append(surrogate.work_estimate)

    errorlist = np.sqrt(np.mean((np.array(predictions) - testvalues_fine) ** 2, axis=1))
    sals_errorlist = np.sqrt(np.mean((np.array(predictions) - testvalues) ** 2, axis=1))

    fig, ax = plt.subplots()

    # Kurven mit Labels
    ax.loglog(work, errorlist, marker="o", linestyle="-", label="total error")
    ax.loglog(work, sals_errorlist, marker="s", linestyle="-", label="pure SALS error")

    # FEM-Referenzlinie (horizontal)
    fem_line = max(float(fem_error), np.finfo(float).tiny)  # log-Skala braucht > 0
    ax.axhline(
        y=fem_line, linestyle="--", linewidth=1.2, label=f"FEM error = {fem_error:.2e}"
    )

    # Achsenbeschriftungen, Titel, Grid
    ax.set_xlabel("Estimated work: N * 2^l")
    ax.set_ylabel("RMSE on test set")
    ax.set_title(
        f"ALS={args.als}, level={args.level}, d={args.d}, min error={np.min(errorlist):.2e})"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Legende
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        Path(args.directory)
        / f"{args.als}{args.rho_option}d{args.d}mesh2^{args.level}.png",
        dpi=200,
    )
    plt.show()
