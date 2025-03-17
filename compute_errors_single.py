import argparse
import os
import numpy as np
from numpy.polynomial.legendre import legval
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


def evaluate(points, components):
    d = len(components)
    leg_sups = np.sqrt(2 * np.arange(components[0].shape[1]) + 1)
    evaluated_legendre = legval(points, np.diag(leg_sups)).T

    result = np.einsum("mi,hij->mj", evaluated_legendre[0], components[0])

    for mode in range(1, d - 1):
        result = np.einsum(
            "mh,mi,hij->mj", result, evaluated_legendre[mode], components[mode]
        )

    return np.einsum("mh,mi,hij->m", result, evaluated_legendre[-1], components[-1])


def plot_results(df, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate duplicate (l_train, N_SLM) entries by taking the mean
    df_grouped = df.groupby(["l_train", "N_SLM"], as_index=False).mean()

    # Compute estimated work
    df_grouped["Estimated Work"] = df_grouped["N_SLM"] * (2 ** df_grouped["l_train"])

    # Sort data for correct plotting order
    df_grouped = df_grouped.sort_values(by=["l_train", "N_SLM"])
    l_train_values = sorted(df_grouped["l_train"].unique())

    ### First Plot: SMSE SLM vs N_SLM ###
    plt.figure(figsize=(15, 10))

    for l in l_train_values:
        subset = df_grouped[df_grouped["l_train"] == l]
        plt.plot(subset["N_SLM"], subset["SMSE SLM"], marker="o", label=f"l_train={l}")

        mean_fem = subset["SMSE FEM"].mean()
        plt.axhline(
            y=mean_fem,
            linestyle="--",
            linewidth=1,
            label=f"Mean SMSE FEM for l_train={l}",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N_SLM")
    plt.ylabel("SMSE SLM")
    plt.title("Log-Log Plot of SMSE SLM vs N_SLM for Different l_train")
    plt.legend(title="l_train", loc="best", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plot_path = os.path.join(output_dir, "training_error_vs_N_slm.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Save the plot
    print(f"Plot saved: {plot_path}")
    plt.close()

    ### Second Plot: SMSE SLM vs Estimated Work ###
    df_grouped = df_grouped.sort_values(by=["l_train", "Estimated Work"])

    plt.figure(figsize=(15, 10))

    for l in l_train_values:
        subset = df_grouped[df_grouped["l_train"] == l]
        plt.plot(
            subset["Estimated Work"],
            subset["SMSE SLM"],
            marker="o",
            label=f"l_train={l}",
        )

        mean_fem = subset["SMSE FEM"].mean()
        plt.axhline(
            y=mean_fem,
            linestyle="--",
            linewidth=1,
            label=f"Mean SMSE FEM for l_train={l}",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Estimated Work (N_SLM * 2^l_train)")
    plt.ylabel("SMSE SLM")
    plt.title("Log-Log Plot of SMSE SLM vs Estimated Work for Different l_train")
    plt.legend(title="l_train", loc="best", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plot_path = os.path.join(output_dir, "training_error_vs_estimated_work.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Save the plot
    print(f"Plot saved: {plot_path}")
    plt.close()


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize ALS components."
    )
    parser.add_argument(
        "--components_file",
        type=str,
        required=True,
        help="Path to ALS components file.",
    )
    parser.add_argument(
        "--samples_file", type=str, required=True, help="Path to samples file."
    )
    parser.add_argument(
        "--errors_file",
        type=str,
        default="errors_single.csv",
        help="Path to errors CSV file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Directory to save plots."
    )

    args = parser.parse_args()

    # Load FEM sample data
    with np.load(args.samples_file) as sample_data:
        points_test = sample_data["points_test"]
        values_test = sample_data["values_test"]
        l_test = sample_data["levels"][-1].item()
        N_test, d_test = points_test.shape

        points_train = sample_data["points_train"]
        values_train = sample_data["values_train"]
        levels_train = sample_data["levels"][:-1].tolist()
        N_fem, d_train = points_train.shape

    # Load tensor train components
    with np.load(args.components_file) as components_data:
        components_dict = defaultdict(lambda: defaultdict(lambda: [None] * d_train))

        for key in components_data.keys():
            try:
                l, N, mode = map(int, key.split("-"))  # Parse "{l}-{N}-{mode}"
                if mode >= d_train:
                    raise ValueError(
                        f"Mode {mode} exceeds expected d_train size {d_train}"
                    )
                components_dict[l][N][mode] = components_data[key]
            except ValueError as e:
                print(f"Skipping invalid key '{key}': {e}")

    # Convert defaultdict to standard dict
    components_dict = {
        l: dict(N_fem_values) for l, N_fem_values in components_dict.items()
    }

    # Load errors file
    df = pd.read_csv(args.errors_file)

    for l in components_dict.keys():
        l_idx = levels_train.index(l)

        for N in components_dict[l].keys():
            error_dict = {
                "l_train": l,
                "d_train": d_train,
                "l_test": l_test,
                "d_test": d_test,
                "N_SLM": N,
                "N_test": N_test,
                "N_FEM": N_fem,
            }

            components = components_dict[l][N]
            values_pred = evaluate(points_test, components)

            error_dict["SMSE SLM"] = np.sqrt(
                np.mean((values_test[-1] - values_pred) ** 2)
            )
            error_dict["SMSE FEM"] = np.sqrt(
                np.mean((values_train[-1] - values_train[l_idx]) ** 2)
            )

            df = pd.concat([df, pd.DataFrame([error_dict])], ignore_index=True)

    plot_results(df, args.output_dir)
    df.to_csv(args.errors_file, index=False)


if __name__ == "__main__":
    main()
