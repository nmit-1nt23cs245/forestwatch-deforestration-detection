import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from test_normalized_dataloader import ForestWatchDataset
from model_resnet50_14ch import ResNet50_14Channel


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8

MODEL_PATH = (
    "data/processed/models/"
    "best_resnet50_14ch.pth"
)

NUM_CLASSES = 2


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
    print("FORESTWATCH - 14-CHANNEL RESNET50 TEST EVALUATION")
    print("=" * 80)

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dataset = ForestWatchDataset(
        "test"
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

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best validation F1: "
        f"{checkpoint.get('best_val_f1', 'unknown')}"
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

            # Probability of class 1 = forest loss
            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    all_labels = np.asarray(
        all_labels
    )

    all_predictions = np.asarray(
        all_predictions
    )

    all_probabilities = np.asarray(
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

    # ROC-AUC requires both classes
    if len(np.unique(all_labels)) == 2:

        roc_auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

        pr_auc = average_precision_score(
            all_labels,
            all_probabilities
        )

    else:

        roc_auc = float("nan")
        pr_auc = float("nan")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

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
        f"Actual No Loss   {tn:7d}   {fp:4d}"
    )

    print(
        f"Actual Loss      {fn:7d}   {tp:4d}"
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

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ TEST EVALUATION COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()