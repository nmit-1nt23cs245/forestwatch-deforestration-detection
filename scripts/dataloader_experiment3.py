import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_DIR = "data/processed/patches"

STATS_PATH = (
    "data/processed/experiment3_channel_stats.json"
)


# ============================================================
# CHANNEL NAMES
# ============================================================

CHANNEL_NAMES = [
    # --------------------------------------------------------
    # Year 1
    # --------------------------------------------------------

    "Year1_B2",
    "Year1_B3",
    "Year1_B4",
    "Year1_B8",
    "Year1_NDVI",
    "Year1_VV",
    "Year1_VH",

    # --------------------------------------------------------
    # Year 2
    # --------------------------------------------------------

    "Year2_B2",
    "Year2_B3",
    "Year2_B4",
    "Year2_B8",
    "Year2_NDVI",
    "Year2_VV",
    "Year2_VH",

    # --------------------------------------------------------
    # Temporal Difference
    #
    # Delta = Year2 - Year1
    # --------------------------------------------------------

    "Delta_B2",
    "Delta_B3",
    "Delta_B4",
    "Delta_B8",
    "Delta_NDVI",
    "Delta_VV",
    "Delta_VH"
]


# ============================================================
# DATASET
# ============================================================

class ForestWatchDatasetExperiment3(Dataset):

    def __init__(
        self,
        split,
        augment=False
    ):

        self.split = split

        # ----------------------------------------------------
        # Augmentation is allowed only for training.
        # Validation and test are never augmented.
        # ----------------------------------------------------

        self.augment = (
            augment and split == "train"
        )

        self.split_dir = os.path.join(
            PATCH_DIR,
            split
        )

        self.metadata_path = os.path.join(
            PATCH_DIR,
            "patch_metadata.csv"
        )

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        metadata = pd.read_csv(
            self.metadata_path
        )

        metadata = metadata[
            metadata["split"] == split
        ].reset_index(
            drop=True
        )

        self.metadata = metadata

        # ----------------------------------------------------
        # Load Experiment 3 channel statistics
        # ----------------------------------------------------

        with open(
            STATS_PATH,
            "r"
        ) as f:

            stats = json.load(f)

        stats_channel_names = stats[
            "channel_names"
        ]

        stats_means = stats[
            "mean"
        ]

        stats_stds = stats[
            "std"
        ]

        # ----------------------------------------------------
        # Create channel -> statistics mapping
        # ----------------------------------------------------

        stats_by_channel = {
            name: {
                "mean": mean,
                "std": std
            }
            for name, mean, std in zip(
                stats_channel_names,
                stats_means,
                stats_stds
            )
        }

        # ----------------------------------------------------
        # Verify all required channels exist
        # ----------------------------------------------------

        missing_channels = [
            name
            for name in CHANNEL_NAMES
            if name not in stats_by_channel
        ]

        if missing_channels:

            raise ValueError(
                "Missing Experiment 3 channel "
                f"statistics: {missing_channels}"
            )

        # ----------------------------------------------------
        # Extract statistics in the exact channel order
        # ----------------------------------------------------

        self.means = np.array(
            [
                stats_by_channel[name]["mean"]
                for name in CHANNEL_NAMES
            ],
            dtype=np.float32
        )

        self.stds = np.array(
            [
                stats_by_channel[name]["std"]
                for name in CHANNEL_NAMES
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Prevent division by zero
        # ----------------------------------------------------

        self.stds[
            self.stds == 0
        ] = 1.0

        # ----------------------------------------------------
        # Final safety check
        # ----------------------------------------------------

        if len(self.means) != 21:

            raise ValueError(
                "Expected 21 channel means, "
                f"got {len(self.means)}"
            )

        if len(self.stds) != 21:

            raise ValueError(
                "Expected 21 channel standard deviations, "
                f"got {len(self.stds)}"
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.metadata
        )

    # ========================================================
    # AUGMENTATION
    # ========================================================

    def apply_augmentation(
        self,
        image
    ):

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if np.random.rand() < 0.5:

            image = np.flip(
                image,
                axis=2
            ).copy()

        # ----------------------------------------------------
        # Vertical flip
        # ----------------------------------------------------

        if np.random.rand() < 0.5:

            image = np.flip(
                image,
                axis=1
            ).copy()

        # ----------------------------------------------------
        # Random 90-degree rotation
        # ----------------------------------------------------

        if np.random.rand() < 0.5:

            k = np.random.randint(
                1,
                4
            )

            image = np.rot90(
                image,
                k=k,
                axes=(1, 2)
            ).copy()

        return image

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        row = self.metadata.iloc[
            index
        ]

        filename = row[
            "filename"
        ]

        label = int(
            row["label"]
        )

        path = os.path.join(
            self.split_dir,
            filename
        )

        # ----------------------------------------------------
        # Load existing 14-channel patch
        # ----------------------------------------------------

        image = np.load(
            path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Verify original patch shape
        # ----------------------------------------------------

        if image.shape != (
            14,
            224,
            224
        ):

            raise ValueError(
                f"Unexpected original patch shape "
                f"{image.shape} in {filename}"
            )

        # ----------------------------------------------------
        # Check original patch for invalid values
        # ----------------------------------------------------

        if not np.isfinite(
            image
        ).all():

            raise ValueError(
                f"NaN or Inf detected in "
                f"{filename}"
            )

        # ----------------------------------------------------
        # Split temporal observations
        #
        # Channels 0-6  = Year 1
        # Channels 7-13 = Year 2
        # ----------------------------------------------------

        year1 = image[
            :7
        ]

        year2 = image[
            7:14
        ]

        # ----------------------------------------------------
        # Calculate temporal difference
        #
        # Delta = Year2 - Year1
        # ----------------------------------------------------

        delta = (
            year2 -
            year1
        )

        # ----------------------------------------------------
        # Construct 21-channel representation
        #
        # 7 channels  : Year 1
        # 7 channels  : Year 2
        # 7 channels  : Year 2 - Year 1
        #
        # Total = 21 channels
        # ----------------------------------------------------

        image = np.concatenate(
            [
                year1,
                year2,
                delta
            ],
            axis=0
        )

        # ----------------------------------------------------
        # Verify final shape
        # ----------------------------------------------------

        if image.shape != (
            21,
            224,
            224
        ):

            raise ValueError(
                f"Unexpected Experiment 3 "
                f"image shape: {image.shape}"
            )

        # ----------------------------------------------------
        # Normalize using training-only statistics
        # ----------------------------------------------------

        image = (
            image -
            self.means[:, None, None]
        ) / (
            self.stds[:, None, None]
        )

        # ----------------------------------------------------
        # Verify normalization did not create invalid values
        # ----------------------------------------------------

        if not np.isfinite(
            image
        ).all():

            raise ValueError(
                f"NaN or Inf detected after "
                f"normalization in {filename}"
            )

        # ----------------------------------------------------
        # Training-only augmentation
        # ----------------------------------------------------

        if self.augment:

            image = self.apply_augmentation(
                image
            )

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        # ----------------------------------------------------
        # Convert label to tensor
        # ----------------------------------------------------

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# BASIC TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "FORESTWATCH - EXPERIMENT 3 DATASET TEST"
    )
    print("=" * 80)

    train_dataset = (
        ForestWatchDatasetExperiment3(
            "train",
            augment=True
        )
    )

    validation_dataset = (
        ForestWatchDatasetExperiment3(
            "validation",
            augment=False
        )
    )

    test_dataset = (
        ForestWatchDatasetExperiment3(
            "test",
            augment=False
        )
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
    # Test one sample
    # --------------------------------------------------------

    image, label = train_dataset[0]

    print(
        f"\nSample image shape : "
        f"{tuple(image.shape)}"
    )

    print(
        f"Sample label       : "
        f"{label.item()}"
    )

    print(
        f"Image dtype        : "
        f"{image.dtype}"
    )

    print(
        f"Image min          : "
        f"{image.min().item():.6f}"
    )

    print(
        f"Image max          : "
        f"{image.max().item():.6f}"
    )

    print(
        f"Image mean         : "
        f"{image.mean().item():.6f}"
    )

    print(
        f"Image std          : "
        f"{image.std().item():.6f}"
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    assert image.shape == (
        21,
        224,
        224
    )

    assert image.dtype == torch.float32

    assert label.dtype == torch.int64

    assert torch.isfinite(
        image
    ).all()

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ EXPERIMENT 3 DATASET TEST PASSED"
    )

    print(
        "=" * 80
    )