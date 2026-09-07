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

from dataloader_experiment3 import (
    ForestWatchDatasetExperiment3
)

from model_resnet50_21ch import (
    ResNet50_21Channel
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8

NUM_WORKERS = 0

MODEL_PATH = (
    "data/processed/models/"
    "experiment3_best_resnet50.pth"
)

RESULTS_DIR = (
    "results/metrics"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "experiment3_metrics.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "experiment3_confusion_matrix.npy"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "FORESTWATCH - EXPERIMENT 3 TEST EVALUATION"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dataset = (
        ForestWatchDatasetExperiment3(
            split="test",
            augment=False
        )
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading Experiment 3 checkpoint..."
    )

    model = ResNet50_21Channel(
        num_classes=2,
        pretrained=True
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation F1: "
        f"{checkpoint['best_val_f1']}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_labels = []

    all_predictions = []

    all_probabilities = []

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
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
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    all_probabilities = np.array(
        all_probabilities
    )

    test_loss = (
        total_loss /
        len(test_dataset)
    )

    # --------------------------------------------------------
    # Classification metrics
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

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    true_negative = cm[0, 0]

    false_positive = cm[0, 1]

    false_negative = cm[1, 0]

    true_positive = cm[1, 1]

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "EXPERIMENT 3 TEST RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTest Loss : {test_loss:.4f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
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
        f"Actual No Loss   "
        f"{true_negative:8d}"
        f"{false_positive:8d}"
    )

    print(
        f"Actual Loss      "
        f"{false_negative:8d}"
        f"{true_positive:8d}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        f"  True Negatives : "
        f"{true_negative}"
    )

    print(
        f"  False Positives: "
        f"{false_positive}"
    )

    print(
        f"  False Negatives: "
        f"{false_negative}"
    )

    print(
        f"  True Positives  : "
        f"{true_positive}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

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
            "21-channel Temporal ResNet50 - Experiment 3",
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(roc_auc, 4),
            round(pr_auc, 4),
            int(true_negative),
            int(false_positive),
            int(false_negative),
            int(true_positive)
        ])

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ EXPERIMENT 3 TEST EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()