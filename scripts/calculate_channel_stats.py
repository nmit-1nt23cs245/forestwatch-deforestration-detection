import os
import csv
import numpy as np
import json


PATCH_FOLDER = "data/processed/patches"
METADATA_PATH = os.path.join(
    PATCH_FOLDER,
    "patch_metadata.csv"
)

OUTPUT_PATH = "data/processed/channel_stats.json"

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

NUM_CHANNELS = 14


print("=" * 80)
print("FORESTWATCH - TRAINING CHANNEL STATISTICS")
print("=" * 80)

# ------------------------------------------------------------
# Load training metadata
# ------------------------------------------------------------

with open(
    METADATA_PATH,
    "r",
    newline=""
) as f:

    reader = csv.DictReader(f)

    train_samples = [
        row
        for row in reader
        if row["split"] == "train"
    ]

print(
    f"\nTraining patches: {len(train_samples)}"
)

if len(train_samples) == 0:
    raise ValueError(
        "No training samples found."
    )


# ------------------------------------------------------------
# Running statistics
#
# We calculate mean/std without loading the entire dataset
# into memory at once.
# ------------------------------------------------------------

sum_values = np.zeros(
    NUM_CHANNELS,
    dtype=np.float64
)

sum_squared = np.zeros(
    NUM_CHANNELS,
    dtype=np.float64
)

pixel_count = np.zeros(
    NUM_CHANNELS,
    dtype=np.int64
)


# ------------------------------------------------------------
# Process training patches
# ------------------------------------------------------------

for index, row in enumerate(
    train_samples,
    start=1
):

    filename = row["filename"]

    filepath = os.path.join(
        PATCH_FOLDER,
        "train",
        filename
    )

    patch = np.load(
        filepath
    ).astype(
        np.float64
    )

    if patch.shape != (
        NUM_CHANNELS,
        224,
        224
    ):

        raise ValueError(
            f"Unexpected shape in {filename}: "
            f"{patch.shape}"
        )

    if not np.isfinite(patch).all():

        raise ValueError(
            f"Non-finite values found in "
            f"{filename}"
        )

    # Flatten spatial dimensions
    pixels = patch.reshape(
        NUM_CHANNELS,
        -1
    )

    sum_values += pixels.sum(
        axis=1
    )

    sum_squared += np.square(
        pixels
    ).sum(
        axis=1
    )

    pixel_count += pixels.shape[1]

    if index % 25 == 0:

        print(
            f"  Processed "
            f"{index}/{len(train_samples)} patches"
        )


# ------------------------------------------------------------
# Calculate statistics
# ------------------------------------------------------------

means = (
    sum_values /
    pixel_count
)

variances = (
    sum_squared /
    pixel_count
) - np.square(means)

# Protect against tiny floating-point negatives
variances = np.maximum(
    variances,
    0
)

stds = np.sqrt(
    variances
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CHANNEL STATISTICS")
print("=" * 80)

for i in range(NUM_CHANNELS):

    print(
        f"{CHANNEL_NAMES[i]:12s} "
        f"Mean = {means[i]:12.6f}   "
        f"Std = {stds[i]:12.6f}"
    )


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

if not np.isfinite(means).all():

    raise ValueError(
        "Invalid mean detected."
    )

if not np.isfinite(stds).all():

    raise ValueError(
        "Invalid standard deviation detected."
    )

if np.any(stds <= 0):

    raise ValueError(
        "One or more channels have zero "
        "standard deviation."
    )


# ------------------------------------------------------------
# Save statistics
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

stats = {
    "channel_names": CHANNEL_NAMES,
    "mean": means.tolist(),
    "std": stds.tolist(),
    "num_training_patches": len(
        train_samples
    ),
    "patch_shape": [
        14,
        224,
        224
    ]
}


with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        stats,
        f,
        indent=4
    )


print(
    f"\nSaved statistics to:"
)

print(
    f"  {OUTPUT_PATH}"
)

print(
    "\n✅ Training channel statistics calculated successfully."
)

print("=" * 80)