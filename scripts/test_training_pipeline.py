import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from test_normalized_dataloader import ForestWatchDataset
from model_resnet50_14ch import ResNet50_14Channel


BATCH_SIZE = 8


def main():

    print("=" * 80)
    print("FORESTWATCH - ONE-BATCH TRAINING PIPELINE TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = ForestWatchDataset(
        "train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet50_14Channel(
        num_classes=2,
        pretrained=True
    )

    model = model.to(device)

    model.train()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Get one real batch
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    images = images.to(device)
    labels = labels.to(device)

    print(
        f"\nInput batch : {tuple(images.shape)}"
    )

    print(
        f"Labels      : {tuple(labels.shape)}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    optimizer.zero_grad()

    outputs = model(
        images
    )

    print(
        f"Model output: {tuple(outputs.shape)}"
    )

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = criterion(
        outputs,
        labels
    )

    print(
        f"Loss        : {loss.item():.6f}"
    )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    gradient_count = 0
    nonzero_gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_count += 1

            if torch.any(
                parameter.grad != 0
            ):

                nonzero_gradient_count += 1

    print(
        f"\nParameters with gradients     : "
        f"{gradient_count}"
    )

    print(
        f"Parameters with nonzero grads : "
        f"{nonzero_gradient_count}"
    )

    if gradient_count == 0:

        raise RuntimeError(
            "No gradients were generated."
        )

    if nonzero_gradient_count == 0:

        raise RuntimeError(
            "All gradients are zero."
        )

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    optimizer.step()

    print(
        "\nOptimizer step: successful"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ ONE-BATCH TRAINING PIPELINE TEST PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()