import torch
import torch.nn as nn

from torch.utils.data import DataLoader

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

LEARNING_RATE = 5e-5


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "FORESTWATCH - EXPERIMENT 3 "
        "ONE-BATCH TRAINING PIPELINE TEST"
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
    # Dataset
    # --------------------------------------------------------

    print(
        "\nLoading Experiment 3 dataset..."
    )

    train_dataset = (
        ForestWatchDatasetExperiment3(
            split="train",
            augment=True
        )
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    images = images.to(
        device
    )

    labels = labels.to(
        device
    )

    print(
        f"\nInput batch : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Labels      : "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Image dtype : "
        f"{images.dtype}"
    )

    print(
        f"Label dtype : "
        f"{labels.dtype}"
    )

    # --------------------------------------------------------
    # Verify input shape
    # --------------------------------------------------------

    assert images.shape == (
        BATCH_SIZE,
        21,
        224,
        224
    )

    assert labels.shape == (
        BATCH_SIZE,
    )

    assert images.dtype == torch.float32

    assert labels.dtype == torch.int64

    # --------------------------------------------------------
    # Verify values
    # --------------------------------------------------------

    print(
        f"Image min   : "
        f"{images.min().item():.6f}"
    )

    print(
        f"Image max   : "
        f"{images.max().item():.6f}"
    )

    print(
        f"Image mean  : "
        f"{images.mean().item():.6f}"
    )

    print(
        f"Image std   : "
        f"{images.std().item():.6f}"
    )

    assert torch.isfinite(
        images
    ).all()

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

    model.train()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    outputs = model(
        images
    )

    print(
        f"\nModel output: "
        f"{tuple(outputs.shape)}"
    )

    assert outputs.shape == (
        BATCH_SIZE,
        2
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        outputs,
        labels
    )

    print(
        f"Loss        : "
        f"{loss.item():.6f}"
    )

    assert torch.isfinite(
        loss
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.001
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    parameters_with_gradients = 0

    parameters_with_nonzero_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            parameters_with_gradients += 1

            if torch.any(
                parameter.grad != 0
            ):

                parameters_with_nonzero_gradients += 1

    print(
        f"\nParameters with gradients     : "
        f"{parameters_with_gradients}"
    )

    print(
        f"Parameters with nonzero grads : "
        f"{parameters_with_nonzero_gradients}"
    )

    assert (
        parameters_with_gradients > 0
    )

    assert (
        parameters_with_nonzero_gradients > 0
    )

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    optimizer.step()

    print(
        "\nOptimizer step: successful"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ EXPERIMENT 3 ONE-BATCH "
        "TRAINING PIPELINE TEST PASSED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()