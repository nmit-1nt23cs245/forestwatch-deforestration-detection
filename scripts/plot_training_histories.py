import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "data/processed/models"
RESULTS_DIR = "results/metrics"


EXPERIMENTS = {
    "Experiment 2": {
        "csv": "experiment2_training_history.csv",
        "output": "experiment2_training_history.png",
        "title": (
            "ForestWatch - 14-Channel ResNet50 "
            "Experiment 2 Training History"
        )
    },

    "Experiment 3": {
        "csv": "experiment3_training_history.csv",
        "output": "experiment3_training_history.png",
        "title": (
            "ForestWatch - 21-Channel ResNet50 "
            "Experiment 3 Training History"
        )
    }
}


# ============================================================
# PLOT FUNCTION
# ============================================================

def create_training_history_plot(
    csv_path,
    output_path,
    title
):

    history = pd.read_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_columns = [
        "epoch",
        "train_f1",
        "val_f1",
        "train_loss",
        "val_loss"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    # --------------------------------------------------------
    # F1 curves
    # --------------------------------------------------------

    ax.plot(
        history["epoch"],
        history["train_f1"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="Training F1"
    )

    ax.plot(
        history["epoch"],
        history["val_f1"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="Validation F1"
    )

    # --------------------------------------------------------
    # Loss curves
    # --------------------------------------------------------

    ax.plot(
        history["epoch"],
        history["train_loss"],
        marker="s",
        linestyle="--",
        linewidth=2,
        markersize=8,
        label="Training Loss"
    )

    ax.plot(
        history["epoch"],
        history["val_loss"],
        marker="s",
        linestyle="--",
        linewidth=2,
        markersize=8,
        label="Validation Loss"
    )

    # --------------------------------------------------------
    # Labels and title
    # --------------------------------------------------------

    ax.set_xlabel(
        "Epoch",
        fontsize=16
    )

    ax.set_ylabel(
        "Score / Loss",
        fontsize=16
    )

    ax.set_title(
        title,
        fontsize=22,
        pad=15
    )

    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------

    ax.set_xticks(
        history["epoch"]
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        fontsize=14,
        loc="upper left"
    )

    # --------------------------------------------------------
    # Tick size
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=13
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "FORESTWATCH - TRAINING HISTORY PLOTS"
    )

    print("=" * 80)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    for experiment, config in EXPERIMENTS.items():

        csv_path = os.path.join(
            MODEL_DIR,
            config["csv"]
        )

        output_path = os.path.join(
            RESULTS_DIR,
            config["output"]
        )

        if not os.path.exists(
            csv_path
        ):

            raise FileNotFoundError(
                f"Training history not found: "
                f"{csv_path}"
            )

        print(
            f"\nGenerating {experiment} plot..."
        )

        create_training_history_plot(
            csv_path=csv_path,
            output_path=output_path,
            title=config["title"]
        )

        print(
            f"  Saved: {output_path}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ TRAINING HISTORY PLOTS CREATED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()