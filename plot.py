import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """Parse CLI arguments for plot configuration."""
    parser = argparse.ArgumentParser(description="Plot error vs. work curves.")

    parser.add_argument(
        "--als",
        type=str,
        default="SALS",
        help="ALS variant used in experiment (SALS or SSALS). Default: SALS",
    )
    parser.add_argument(
        "-lmax",
        "--maxlevel",
        type=int,
        default=8,
        help="Maximal training FEM mesh level used in experiment. Default: 8",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Optional: explicit filename to load (overrides als/maxlevel naming convention).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine filename to load
    if args.filename:
        fn = args.filename
    else:
        fn = f"errorvswork{args.als}level{args.maxlevel}.npz"

    # ------------------------
    # Load data from .npz file
    # ------------------------
    data = np.load(fn)
    work = data["work"]  # shape (n_levels, n_sample_sizes)
    er = data["er"]  # same shape as work
    femerror = float(data["femerror"])  # scalar baseline error

    # Sanity check: work and error arrays must align
    if er.shape != work.shape:
        raise ValueError(f"Shape mismatch: er{er.shape} vs work{work.shape}")

    # ------------------------
    # Plotting
    # ------------------------
    fig, ax = plt.subplots()

    # Loop over multilevel depths (rows of er/work)
    for L in range(er.shape[0]):
        x = work[L]
        y = er[L]

        # Mask out invalid or zero entries (for log-log plotting)
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            ax.plot(
                x[mask],
                y[mask],
                marker="o",
                linestyle="-",
                label=f"L = {L}",  # L=0 is single-level, L>=1 are multilevel depths
            )

    # Optional: horizontal reference line for FEM baseline error
    # Uncomment if you want to see where the FEM discretization sits
    ax.axhline(femerror, linestyle="--", linewidth=1, color="gray", label="FEM error")

    # Log-log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Labels and grid
    ax.set_xlabel("Work (model units)")
    ax.set_ylabel("Error (RMSE vs. reference)")
    ax.grid(True, which="both", linestyle=":")

    plt.title(fn)

    # Legend and layout
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
