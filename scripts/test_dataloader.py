import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PATCH_FOLDER = "data/processed/patches"
METADATA_PATH = os.path.join(
    PATCH_FOLDER,
    "patch_metadata.csv"
)


class ForestWatchDataset(Dataset):

    def __init__(self, split):

        self.split = split

        self.patch_folder = os.path.join(
            PATCH_FOLDER,
            split
        )

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        with open(
            METADATA_PATH,
            "r",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            self.samples = [
                row
                for row in reader
                if row["split"] == split
            ]

        if len(self.samples) == 0:

            raise ValueError(
                f"No samples found for split: {split}"
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        row = self.samples[index]

        filename = row["filename"]

        label = int(
            row["label"]
        )

        filepath = os.path.join(
            self.patch_folder,
            filename
        )

        patch = np.load(
            filepath
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if patch.shape != (
            14,
            224,
            224
        ):

            raise ValueError(
                f"Unexpected patch shape: "
                f"{patch.shape}"
            )

        if not np.isfinite(
            patch
        ).all():

            raise ValueError(
                f"Non-finite values found in "
                f"{filename}"
            )

        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # ----------------------------------------------------

        patch = torch.from_numpy(
            patch.astype(
                np.float32
            )
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return patch, label


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FORESTWATCH - DATALOADER TEST")
    print("=" * 70)

    train_dataset = ForestWatchDataset(
        "train"
    )

    validation_dataset = ForestWatchDataset(
        "validation"
    )

    test_dataset = ForestWatchDataset(
        "test"
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
    # Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Test one training batch
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    print(
        f"\nBatch image shape : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Batch label shape : "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Image dtype       : "
        f"{images.dtype}"
    )

    print(
        f"Label dtype       : "
        f"{labels.dtype}"
    )

    print(
        f"Labels            : "
        f"{labels.tolist()}"
    )

    print(
        f"\nImage min         : "
        f"{images.min().item():.6f}"
    )

    print(
        f"Image max         : "
        f"{images.max().item():.6f}"
    )

    print(
        f"Image mean        : "
        f"{images.mean().item():.6f}"
    )

    print(
        f"Image std         : "
        f"{images.std().item():.6f}"
    )

    print(
        "\n✅ DataLoader test successful."
    )

    print("=" * 70)