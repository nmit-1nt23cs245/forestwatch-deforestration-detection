import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

METRICS_DIR = "results/metrics"


MATRICES = {
    "Experiment 2": {
        "input": "experiment2_confusion_matrix.npy",
        "output": "experiment2_confusion_matrix.png"
    },

    "Experiment 3": {
        "input": "experiment3_confusion_matrix.npy",
        "output": "experiment3_confusion_matrix.png"
    }
}


# ============================================================
# PLOT FUNCTION
# ============================================================

def create_confusion_matrix_plot(
    matrix,
    title,
    output_path
):

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest"
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[
            "No Loss",
            "Loss"
        ],
        yticklabels=[
            "No Loss",
            "Loss"
        ],
        xlabel="Predicted Class",
        ylabel="Actual Class",
        title=title
    )

    # --------------------------------------------------------
    # Display values inside cells
    # --------------------------------------------------------

    threshold = (
        matrix.max() / 2.0
    )

    for row in range(
        matrix.shape[0]
    ):

        for col in range(
            matrix.shape[1]
        ):

            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, col] > threshold
                    else "black"
                ),
                fontsize=16,
                fontweight="bold"
            )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_ylim(
        len(matrix) - 0.5,
        -0.5
    )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

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
        "FORESTWATCH - CONFUSION MATRIX PLOTS"
    )
    print("=" * 80)

    os.makedirs(
        METRICS_DIR,
        exist_ok=True
    )

    for experiment, config in MATRICES.items():

        input_path = os.path.join(
            METRICS_DIR,
            config["input"]
        )

        output_path = os.path.join(
            METRICS_DIR,
            config["output"]
        )

        # ----------------------------------------------------
        # Load matrix
        # ----------------------------------------------------

        if not os.path.exists(
            input_path
        ):

            raise FileNotFoundError(
                f"Missing confusion matrix: "
                f"{input_path}"
            )

        matrix = np.load(
            input_path
        )

        # ----------------------------------------------------
        # Verify matrix
        # ----------------------------------------------------

        if matrix.shape != (
            2,
            2
        ):

            raise ValueError(
                f"Unexpected confusion matrix "
                f"shape: {matrix.shape}"
            )

        print(
            f"\n{experiment}"
        )

        print(
            "Matrix:"
        )

        print(
            matrix
        )

        # ----------------------------------------------------
        # Create PNG
        # ----------------------------------------------------

        create_confusion_matrix_plot(
            matrix=matrix,
            title=(
                f"ForestWatch - "
                f"{experiment} Confusion Matrix"
            ),
            output_path=output_path
        )

        print(
            f"Saved: {output_path}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ CONFUSION MATRIX PLOTS CREATED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()