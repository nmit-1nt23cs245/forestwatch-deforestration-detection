import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

METRICS_DIR = "results/metrics"

INPUT_PATH = os.path.join(
    METRICS_DIR,
    "experiment_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_results():

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Missing comparison file: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = [
        "experiment",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_positive",
        "false_negative"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    return df


# ============================================================
# CHART 1 - CLASSIFICATION METRICS
# ============================================================

def plot_classification_metrics(df):

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    x = range(
        len(df)
    )

    width = 0.18

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    for index, metric in enumerate(
        metrics
    ):

        values = df[metric]

        positions = [
            value
            + (index - 1.5) * width
            for value in x
        ]

        ax.bar(
            positions,
            values,
            width=width,
            label=labels[index]
        )

        for position, value in zip(
            positions,
            values
        ):

            ax.text(
                position,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        [
            "Baseline",
            "Experiment 2",
            "Experiment 3"
        ]
    )

    ax.set_ylabel(
        "Score",
        fontsize=15
    )

    ax.set_xlabel(
        "Experiment",
        fontsize=15
    )

    ax.set_title(
        "ForestWatch - Classification Performance Comparison",
        fontsize=20,
        pad=15
    )

    ax.set_ylim(
        0,
        1.10
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend(
        fontsize=12
    )

    fig.tight_layout()

    output = os.path.join(
        METRICS_DIR,
        "experiment_performance_comparison.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# CHART 2 - AUC METRICS
# ============================================================

def plot_auc_metrics(df):

    metrics = [
        "roc_auc",
        "pr_auc"
    ]

    labels = [
        "ROC-AUC",
        "PR-AUC"
    ]

    x = range(
        len(df)
    )

    width = 0.32

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    for index, metric in enumerate(
        metrics
    ):

        positions = [
            value
            + (index - 0.5) * width
            for value in x
        ]

        values = df[metric]

        ax.bar(
            positions,
            values,
            width=width,
            label=labels[index]
        )

        for position, value in zip(
            positions,
            values
        ):

            ax.text(
                position,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=11
            )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        [
            "Baseline",
            "Experiment 2",
            "Experiment 3"
        ]
    )

    ax.set_ylabel(
        "AUC Score",
        fontsize=15
    )

    ax.set_xlabel(
        "Experiment",
        fontsize=15
    )

    ax.set_title(
        "ForestWatch - ROC-AUC and PR-AUC Comparison",
        fontsize=20,
        pad=15
    )

    ax.set_ylim(
        0,
        1.10
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend(
        fontsize=12
    )

    fig.tight_layout()

    output = os.path.join(
        METRICS_DIR,
        "experiment_auc_comparison.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# CHART 3 - ERROR COMPARISON
# ============================================================

def plot_error_metrics(df):

    metrics = [
        "false_positive",
        "false_negative"
    ]

    labels = [
        "False Positives",
        "False Negatives"
    ]

    x = range(
        len(df)
    )

    width = 0.32

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    for index, metric in enumerate(
        metrics
    ):

        positions = [
            value
            + (index - 0.5) * width
            for value in x
        ]

        values = df[metric]

        ax.bar(
            positions,
            values,
            width=width,
            label=labels[index]
        )

        for position, value in zip(
            positions,
            values
        ):

            ax.text(
                position,
                value + 0.5,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=11
            )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        [
            "Baseline",
            "Experiment 2",
            "Experiment 3"
        ]
    )

    ax.set_ylabel(
        "Number of Samples",
        fontsize=15
    )

    ax.set_xlabel(
        "Experiment",
        fontsize=15
    )

    ax.set_title(
        "ForestWatch - Classification Error Comparison",
        fontsize=20,
        pad=15
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend(
        fontsize=12
    )

    fig.tight_layout()

    output = os.path.join(
        METRICS_DIR,
        "experiment_error_comparison.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "FORESTWATCH - EXPERIMENT RESULTS VISUALIZATION"
    )

    print("=" * 80)

    df = load_results()

    print(
        "\nGenerating classification metrics chart..."
    )

    plot_classification_metrics(
        df
    )

    print(
        "\nGenerating AUC comparison chart..."
    )

    plot_auc_metrics(
        df
    )

    print(
        "\nGenerating error comparison chart..."
    )

    plot_error_metrics(
        df
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ EXPERIMENT RESULT CHARTS CREATED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()