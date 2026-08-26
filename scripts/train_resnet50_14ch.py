import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from test_normalized_dataloader import ForestWatchDataset
from model_resnet50_14ch import ResNet50_14Channel


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 8

WARMUP_EPOCHS = 3
FINETUNE_EPOCHS = 20

WARMUP_LR = 1e-4
FINETUNE_LR = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 5

NUM_CLASSES = 2

OUTPUT_DIR = "data/processed/models"

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_resnet50_14ch.pth"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions
):

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            labels,
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
    optimizer
):

    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

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
            loss.item() *
            images.size(0)
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

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() *
                images.size(0)
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

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return epoch_loss, metrics


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    fieldnames = [
        "epoch",
        "stage",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_precision",
        "train_recall",
        "train_f1",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1"
    ]

    with open(
        HISTORY_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directory before saving checkpoints
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("FORESTWATCH - 14-CHANNEL RESNET50 TRAINING")
    print("=" * 80)

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = ForestWatchDataset(
        "train"
    )

    validation_dataset = ForestWatchDataset(
        "validation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
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
    # Model
    # --------------------------------------------------------

    print("\nLoading ResNet50...")

    model = ResNet50_14Channel(
        num_classes=NUM_CLASSES,
        pretrained=True
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    history = []

    best_val_f1 = -1.0

    epochs_without_improvement = 0

    global_epoch = 0

    # ========================================================
    # STAGE 1 — WARMUP
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 1 - WARMUP")
    print("=" * 80)

    # Freeze ResNet body
    for parameter in model.model.parameters():
        parameter.requires_grad = False

    # Train the new first convolution
    for parameter in model.model.conv1.parameters():
        parameter.requires_grad = True

    # Train classifier
    for parameter in model.model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=WARMUP_LR,
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(
        WARMUP_EPOCHS
    ):

        global_epoch += 1

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_metrics = validate(
            model,
            validation_loader,
            criterion
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {global_epoch:02d} "
            f"[Warmup]"
        )

        print(
            f"LR={current_lr:.2e} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train F1={train_metrics['f1']:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val F1={val_metrics['f1']:.4f}"
        )

        history.append({
            "epoch": global_epoch,
            "stage": "warmup",
            "learning_rate": current_lr,
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

            best_val_f1 = val_metrics["f1"]

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": best_val_f1
                },
                BEST_MODEL_PATH
            )

            print(
                "  ✅ New best model saved."
            )

        else:

            epochs_without_improvement += 1

    # ========================================================
    # STAGE 2 — FULL FINE-TUNING
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 2 - FULL FINE-TUNING")
    print("=" * 80)

    for parameter in model.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    for epoch in range(
        FINETUNE_EPOCHS
    ):

        global_epoch += 1

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_metrics = validate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(
            val_metrics["f1"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {global_epoch:02d} "
            f"[Fine-tune]"
        )

        print(
            f"LR={current_lr:.2e} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train F1={train_metrics['f1']:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val F1={val_metrics['f1']:.4f}"
        )

        print(
            f"  Val Accuracy={val_metrics['accuracy']:.4f} | "
            f"Precision={val_metrics['precision']:.4f} | "
            f"Recall={val_metrics['recall']:.4f}"
        )

        history.append({
            "epoch": global_epoch,
            "stage": "finetune",
            "learning_rate": current_lr,
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

            best_val_f1 = val_metrics["f1"]

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": best_val_f1
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

        save_history(
            history
        )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\n🛑 Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    save_history(
        history
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"\nBest validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Best model:"
    )

    print(
        f"  {BEST_MODEL_PATH}"
    )

    print(
        f"\nTraining history:"
    )

    print(
        f"  {HISTORY_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()