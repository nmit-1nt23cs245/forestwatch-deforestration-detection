import torch

from torch.utils.data import DataLoader

from dataloader_experiment2 import (
    ForestWatchDatasetExperiment2
)


BATCH_SIZE = 8


def main():

    print("=" * 80)
    print("FORESTWATCH - EXPERIMENT 2 DATALOADER TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset = ForestWatchDatasetExperiment2(
        "train",
        augment=True
    )

    validation_dataset = ForestWatchDatasetExperiment2(
        "validation",
        augment=False
    )

    test_dataset = ForestWatchDatasetExperiment2(
        "test",
        augment=False
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"\nTrain samples      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_dataset)}"
    )

    print(
        f"Test samples       : "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Test training batch
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    print(
        f"\nTraining batch image shape : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Training batch label shape : "
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

    print(
        f"Image min   : "
        f"{images.min().item():.6f}"
    )

    print(
        f"Image max   : "
        f"{images.max().item():.6f}"
    )

    # --------------------------------------------------------
    # Verify validation
    # --------------------------------------------------------

    val_images, val_labels = next(
        iter(validation_loader)
    )

    print(
        f"\nValidation batch shape : "
        f"{tuple(val_images.shape)}"
    )

    # --------------------------------------------------------
    # Verify test
    # --------------------------------------------------------

    test_images, test_labels = next(
        iter(test_loader)
    )

    print(
        f"Test batch shape       : "
        f"{tuple(test_images.shape)}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert images.shape == (
        BATCH_SIZE,
        14,
        224,
        224
    )

    assert labels.shape == (
        BATCH_SIZE,
    )

    assert val_images.shape[1:] == (
        14,
        224,
        224
    )

    assert test_images.shape[1:] == (
        14,
        224,
        224
    )

    assert not torch.isnan(
        images
    ).any()

    assert not torch.isnan(
        val_images
    ).any()

    assert not torch.isnan(
        test_images
    ).any()

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ EXPERIMENT 2 DATALOADER TEST PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()