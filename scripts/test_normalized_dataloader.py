import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PATCH_FOLDER = "data/processed/patches"

METADATA_PATH = os.path.join(
    PATCH_FOLDER,
    "patch_metadata.csv"
)

STATS_PATH = (
    "data/processed/channel_stats.json"
)


class ForestWatchDataset(Dataset):

    def __init__(
        self,
        split,
        stats_path=STATS_PATH
    ):

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
                f"No samples found for split: "
                f"{split}"
            )

        # ----------------------------------------------------
        # Load training statistics
        # ----------------------------------------------------

        with open(
            stats_path,
            "r"
        ) as f:

            stats = json.load(f)

        self.mean = np.asarray(
            stats["mean"],
            dtype=np.float32
        ).reshape(
            14,
            1,
            1
        )

        self.std = np.asarray(
            stats["std"],
            dtype=np.float32
        ).reshape(
            14,
            1,
            1
        )

        if self.mean.shape != (
            14,
            1,
            1
        ):

            raise ValueError(
                "Invalid channel mean shape."
            )

        if self.std.shape != (
            14,
            1,
            1
        ):

            raise ValueError(
                "Invalid channel std shape."
            )

        if np.any(
            self.std <= 0
        ):

            raise ValueError(
                "One or more channel std values "
                "are <= 0."
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
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Validate raw patch
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
        # Channel-wise normalization
        #
        # Statistics were calculated ONLY from training data.
        # ----------------------------------------------------

        patch = (
            patch - self.mean
        ) / self.std

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not np.isfinite(
            patch
        ).all():

            raise ValueError(
                f"Non-finite values after "
                f"normalization: {filename}"
            )

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        patch = torch.from_numpy(
            patch
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return patch, label


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("FORESTWATCH - NORMALIZED DATALOADER TEST")
    print("=" * 80)

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
    # Test training batch
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
        f"\nOverall normalized mean : "
        f"{images.mean().item():.6f}"
    )

    print(
        f"Overall normalized std  : "
        f"{images.std().item():.6f}"
    )

    print(
        f"Overall normalized min  : "
        f"{images.min().item():.6f}"
    )

    print(
        f"Overall normalized max  : "
        f"{images.max().item():.6f}"
    )

    # --------------------------------------------------------
    # Check channel means/stds for this batch
    # --------------------------------------------------------

    channel_means = images.mean(
        dim=(0, 2, 3)
    )

    channel_stds = images.std(
        dim=(0, 2, 3)
    )

    print(
        "\nCHANNEL STATISTICS FOR TEST BATCH"
    )

    print("-" * 80)

    for i in range(14):

        print(
            f"Channel {i + 1:2d}: "
            f"mean={channel_means[i].item():8.4f}, "
            f"std={channel_stds[i].item():8.4f}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ NORMALIZED DATALOADER TEST PASSED"
    )

    print("=" * 80)