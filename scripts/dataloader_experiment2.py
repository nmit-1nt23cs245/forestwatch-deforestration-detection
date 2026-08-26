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
    "data/processed/channel_stats.json"
)

CHANNEL_NAMES = [
    "Year1_B2",
    "Year1_B3",
    "Year1_B4",
    "Year1_B8",
    "Year1_NDVI",
    "Year1_VV",
    "Year1_VH",
    "Year2_B2",
    "Year2_B3",
    "Year2_B4",
    "Year2_B8",
    "Year2_NDVI",
    "Year2_VV",
    "Year2_VH"
]


# ============================================================
# DATASET
# ============================================================

class ForestWatchDatasetExperiment2(Dataset):

    def __init__(
        self,
        split,
        augment=False
    ):

        self.split = split

        # Augmentation is allowed only for training
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
        # Load channel statistics
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
        # Extract statistics in correct channel order
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

        # Prevent division by zero
        self.stds[
            self.stds == 0
        ] = 1.0

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
        # Load patch
        # ----------------------------------------------------

        image = np.load(
            path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image = (
            image -
            self.means[:, None, None]
        ) / (
            self.stds[:, None, None]
        )

        # ----------------------------------------------------
        # Training-only augmentation
        # ----------------------------------------------------

        if self.augment:

            image = self.apply_augmentation(
                image
            )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label