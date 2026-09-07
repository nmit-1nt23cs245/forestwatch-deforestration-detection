import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
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

SEED = 42

BATCH_SIZE = 8

NUM_WORKERS = 0

WARMUP_EPOCHS = 3

MAX_EPOCHS = 30

WARMUP_LR = 5e-5

FINE_TUNE_LR = 5e-6

WEIGHT_DECAY = 1e-3

PATIENCE = 5

MODEL_DIR = (
    "data/processed/models"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "experiment3_best_resnet50.pth"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "experiment3_training_history.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    predictions
):

    return {
        "accuracy": accuracy_score(
            targets,
            predictions
        ),

        "precision": precision_score(
            targets,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            targets,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            targets,
            predictions,
            zero_division=0
        )
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_targets = []

    all_predictions = []

    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            torch.argmax(
                outputs,
                dim=1
            )
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )

    return (
        epoch_loss,
        metrics
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_targets = []

    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

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

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )

    return (
        epoch_loss,
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    print("=" * 80)

    print(
        "FORESTWATCH - EXPERIMENT 3"
    )

    print(
        "21-CHANNEL TEMPORAL RESNET50"
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
    # Configuration
    # --------------------------------------------------------

    print(
        "\nExperiment 3 configuration:"
    )

    print(
        "  Input channels       : 21"
    )

    print(
        "  Year 1 channels      : 7"
    )

    print(
        "  Year 2 channels      : 7"
    )

    print(
        "  Temporal delta       : 7"
    )

    print(
        "  Training augmentation:"
    )

    print(
        "    horizontal flip"
    )

    print(
        "    vertical flip"
    )

    print(
        "    90-degree rotation"
    )

    print(
        f"  Warmup LR             : "
        f"{WARMUP_LR}"
    )

    print(
        f"  Fine-tune LR          : "
        f"{FINE_TUNE_LR}"
    )

    print(
        f"  Weight decay          : "
        f"{WEIGHT_DECAY}"
    )

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    train_dataset = (
        ForestWatchDatasetExperiment3(
            split="train",
            augment=True
        )
    )

    validation_dataset = (
        ForestWatchDatasetExperiment3(
            split="validation",
            augment=False
        )
    )

    print(
        f"Train samples      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading 21-channel ResNet50..."
    )

    model = ResNet50_21Channel(
        num_classes=2,
        pretrained=True
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STAGE 1 - TEMPORAL WARMUP"
    )

    print(
        "=" * 80
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=WARMUP_LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_val_f1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for epoch in range(
        1,
        WARMUP_EPOCHS + 1
    ):

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss, val_metrics = (
            validate(
                model,
                validation_loader,
                criterion,
                device
            )
        )

        print(
            f"\nEpoch {epoch:02d} [Warmup]"
        )

        print(
            f"LR={WARMUP_LR:.2e} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train F1={train_metrics['f1']:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val F1={val_metrics['f1']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "stage": "Warmup",
            "learning_rate": WARMUP_LR,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"]
        })

        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "best_val_f1": (
                        best_val_f1
                    )
                },
                BEST_MODEL_PATH
            )

            print(
                "  ✅ New best model saved."
            )

        else:

            epochs_without_improvement += 1

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STAGE 2 - TEMPORAL FULL FINE-TUNING"
    )

    print(
        "=" * 80
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINE_TUNE_LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Fine-tuning
    # --------------------------------------------------------

    for epoch in range(
        WARMUP_EPOCHS + 1,
        MAX_EPOCHS + 1
    ):

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss, val_metrics = (
            validate(
                model,
                validation_loader,
                criterion,
                device
            )
        )

        print(
            f"\nEpoch {epoch:02d} "
            f"[Fine-tune]"
        )

        print(
            f"LR={FINE_TUNE_LR:.2e} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train F1={train_metrics['f1']:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val F1={val_metrics['f1']:.4f}"
        )

        print(
            f"  Val Accuracy="
            f"{val_metrics['accuracy']:.4f} | "
            f"Precision="
            f"{val_metrics['precision']:.4f} | "
            f"Recall="
            f"{val_metrics['recall']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "stage": "Fine-tune",
            "learning_rate": FINE_TUNE_LR,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"]
        })

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "best_val_f1": (
                        best_val_f1
                    )
                },
                BEST_MODEL_PATH
            )

            print(
                "  ✅ New best model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  No improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\n🛑 Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "EXPERIMENT 3 TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nBest validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        "\nBest model:"
    )

    print(
        f"  {BEST_MODEL_PATH}"
    )

    print(
        "\nTraining history:"
    )

    print(
        f"  {HISTORY_PATH}"
    )

    print(
        "\n" + "=" * 80
    )


if __name__ == "__main__":

    main()