import os
import numpy as np


PATCH_FOLDER = "data/processed/patches"

SPLITS = [
    "train",
    "validation",
    "test"
]

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


print("=" * 80)
print("FORESTWATCH - NaN CHANNEL ANALYSIS")
print("=" * 80)


channel_nan_counts = np.zeros(14, dtype=int)
channel_total_counts = np.zeros(14, dtype=int)

files_with_nan = []


for split in SPLITS:

    folder = os.path.join(
        PATCH_FOLDER,
        split
    )

    print(f"\n{split.upper()}")

    for filename in sorted(
        os.listdir(folder)
    ):

        if not filename.endswith(".npy"):
            continue

        filepath = os.path.join(
            folder,
            filename
        )

        patch = np.load(filepath)

        nan_mask = np.isnan(patch)

        if not nan_mask.any():
            continue

        files_with_nan.append(
            (split, filename)
        )

        print(
            f"\n  {filename}"
        )

        for channel in range(14):

            nan_count = np.sum(
                nan_mask[channel]
            )

            total_count = patch[channel].size

            channel_total_counts[channel] += total_count
            channel_nan_counts[channel] += nan_count

            if nan_count > 0:

                percentage = (
                    nan_count /
                    total_count
                ) * 100

                print(
                    f"    {CHANNEL_NAMES[channel]:12s}: "
                    f"{nan_count:,} NaN "
                    f"({percentage:.4f}%)"
                )


print("\n" + "=" * 80)
print("CHANNEL-LEVEL SUMMARY")
print("=" * 80)

for channel in range(14):

    nan_count = channel_nan_counts[channel]
    total_count = channel_total_counts[channel]

    if total_count > 0:

        percentage = (
            nan_count /
            total_count
        ) * 100

    else:

        percentage = 0

    print(
        f"{CHANNEL_NAMES[channel]:12s}: "
        f"{nan_count:,} NaN values "
        f"({percentage:.6f}%)"
    )


print("\n" + "=" * 80)

print(
    f"Files containing NaN: "
    f"{len(files_with_nan)}"
)

print("=" * 80)