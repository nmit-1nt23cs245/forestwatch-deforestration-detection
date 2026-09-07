import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

METRICS_DIR = "results/metrics"

OUTPUT_PATH = os.path.join(
    METRICS_DIR,
    "experiment_comparison.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FORESTWATCH - EXPERIMENT COMPARISON")
    print("=" * 80)

    files = [
        "baseline_14ch_metrics.csv",
        "experiment2_metrics.csv",
        "experiment3_metrics.csv"
    ]

    dataframes = []

    for filename in files:

        path = os.path.join(
            METRICS_DIR,
            filename
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing metrics file: {path}"
            )

        df = pd.read_csv(
            path
        )

        dataframes.append(df)

    # --------------------------------------------------------
    # Combine experiment results
    # --------------------------------------------------------

    comparison = pd.concat(
        dataframes,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print(
        "\nExperiment comparison:"
    )

    print(
        comparison[
            [
                "experiment",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved comparison:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ EXPERIMENT COMPARISON CREATED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()