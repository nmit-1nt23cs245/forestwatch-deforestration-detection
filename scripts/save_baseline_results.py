import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# PATHS
# ============================================================

HISTORY_PATH = (
    "data/processed/models/"
    "training_history.csv"
)

OUTPUT_DIR = "results/metrics"

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "baseline_14ch_metrics.csv"
)

CONFUSION_PATH = os.path.join(
    OUTPUT_DIR,
    "baseline_14ch_confusion_matrix.png"
)

HISTORY_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "baseline_14ch_training_history.png"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# BASELINE TEST RESULTS
# ============================================================

metrics = {
    "experiment": "14-channel Temporal ResNet50",
    "accuracy": 0.6429,
    "precision": 0.6420,
    "recall": 0.8254,
    "f1": 0.7222,
    "roc_auc": 0.6566,
    "pr_auc": 0.6804,
    "true_negative": 20,
    "false_positive": 29,
    "false_negative": 11,
    "true_positive": 52
}


# ============================================================
# SAVE METRICS CSV
# ============================================================

with open(
    METRICS_PATH,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(metrics.keys())
    )

    writer.writeheader()

    writer.writerow(
        metrics
    )


print(
    f"Saved metrics:"
)

print(
    f"  {METRICS_PATH}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = np.array([
    [20, 29],
    [11, 52]
])

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=[
        "No Forest Loss",
        "Forest Loss"
    ]
)

display.plot(
    values_format="d"
)

plt.title(
    "ForestWatch - 14-Channel ResNet50\n"
    "Baseline Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved confusion matrix:"
)

print(
    f"  {CONFUSION_PATH}"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

epochs = []
train_loss = []
val_loss = []
train_f1 = []
val_f1 = []

with open(
    HISTORY_PATH,
    "r"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        epochs.append(
            int(row["epoch"])
        )

        train_loss.append(
            float(row["train_loss"])
        )

        val_loss.append(
            float(row["val_loss"])
        )

        train_f1.append(
            float(row["train_f1"])
        )

        val_f1.append(
            float(row["val_f1"])
        )


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure()

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "ForestWatch - Baseline Training and Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

# Save temporarily as part of combined history plot later


# ============================================================
# F1 CURVE
# ============================================================

plt.figure()

plt.plot(
    epochs,
    train_f1,
    marker="o",
    label="Training F1"
)

plt.plot(
    epochs,
    val_f1,
    marker="o",
    label="Validation F1"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "F1-score"
)

plt.title(
    "ForestWatch - Baseline Training and Validation F1"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

# ============================================================
# COMBINED HISTORY PLOT
# ============================================================

plt.figure()

plt.plot(
    epochs,
    train_f1,
    marker="o",
    label="Training F1"
)

plt.plot(
    epochs,
    val_f1,
    marker="o",
    label="Validation F1"
)

plt.plot(
    epochs,
    train_loss,
    marker="s",
    linestyle="--",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="s",
    linestyle="--",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Score / Loss"
)

plt.title(
    "ForestWatch - 14-Channel ResNet50 Baseline Training History"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    HISTORY_PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved training history:"
)

print(
    f"  {HISTORY_PLOT_PATH}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("BASELINE RESULTS SAVED")
print("=" * 80)

print(
    "\nExperiment: 14-channel Temporal ResNet50"
)

print(
    "Test F1   : 0.7222"
)

print(
    "ROC-AUC   : 0.6566"
)

print(
    "PR-AUC    : 0.6804"
)

print(
    "\n✅ Baseline results recorded successfully."
)

print("=" * 80)