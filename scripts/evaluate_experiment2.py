import os
import csv
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from dataloader_experiment2 import (
    ForestWatchDatasetExperiment2
)

from model_resnet50_14ch import (
    ResNet50_14Channel
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8

NUM_CLASSES = 2

MODEL_PATH = (
    "data/processed/models/"
    "experiment2_best_resnet50.pth"
)

OUTPUT_DIR = (
    "results/metrics"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "experiment2_metrics.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "experiment2_confusion_matrix.npy"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "FORESTWATCH - EXPERIMENT 2 TEST EVALUATION"
    )
    print("=" * 80)

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dataset = (
        ForestWatchDatasetExperiment2(
            "test",
            augment=False
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading Experiment 2 checkpoint..."
    )

    model = ResNet50_14Channel(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    best_val_f1 = checkpoint.get(
        "best_val_f1",
        "unknown"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_val_f1}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(
                images
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    all_probabilities = np.array(
        all_probabilities
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    pr_auc = average_precision_score(
        all_labels,
        all_probabilities
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "EXPERIMENT 2 TEST RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print(
        "                 Predicted"
    )

    print(
        "                 No Loss   Loss"
    )

    print(
        f"Actual No Loss      "
        f"{tn:3d}      {fp:3d}"
    )

    print(
        f"Actual Loss         "
        f"{fn:3d}      {tp:3d}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        f"  True Negatives : {tn}"
    )

    print(
        f"  False Positives: {fp}"
    )

    print(
        f"  False Negatives: {fn}"
    )

    print(
        f"  True Positives  : {tp}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        METRICS_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "experiment",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive"
        ])

        writer.writerow([
            "14-channel Temporal ResNet50 - Experiment 2",
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(roc_auc, 4),
            round(pr_auc, 4),
            int(tn),
            int(fp),
            int(fn),
            int(tp)
        ])

    np.save(
        CONFUSION_MATRIX_PATH,
        cm
    )

    print(
        "\nSaved metrics:"
    )

    print(
        f"  {METRICS_PATH}"
    )

    print(
        "\nSaved confusion matrix:"
    )

    print(
        f"  {CONFUSION_MATRIX_PATH}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ EXPERIMENT 2 TEST EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()