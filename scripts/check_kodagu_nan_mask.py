import os
import numpy as np
import rasterio


DATA_FOLDER = "data/processed/normalized"
MASK_FOLDER = "data/processed/masks"

IMAGE_FILES = [
    "Kodagu_2020_7channel.tif",
    "Kodagu_2022_7channel.tif",
    "Kodagu_2024_7channel.tif"
]

MASK_FILE = "Kodagu_valid_mask.tif"


print("=" * 80)
print("FORESTWATCH - KODAGU NaN / VALID MASK CHECK")
print("=" * 80)


# ============================================================
# Load valid mask
# ============================================================

mask_path = os.path.join(
    MASK_FOLDER,
    MASK_FILE
)

with rasterio.open(mask_path) as src:

    valid_mask = src.read(1)

    print("\nVALID MASK")
    print("-" * 80)

    print(
        f"Size       : {src.width} x {src.height}"
    )

    print(
        f"Data type  : {src.dtypes[0]}"
    )

    print(
        f"Unique vals: {np.unique(valid_mask)}"
    )

    print(
        f"Valid pixels   : "
        f"{np.sum(valid_mask == 1):,}"
    )

    print(
        f"Invalid pixels : "
        f"{np.sum(valid_mask == 0):,}"
    )


# ============================================================
# Check every optical image
# ============================================================

for filename in IMAGE_FILES:

    image_path = os.path.join(
        DATA_FOLDER,
        filename
    )

    print("\n" + "=" * 80)

    print(filename)

    print("=" * 80)

    with rasterio.open(image_path) as src:

        optical = src.read(
            [1, 2, 3, 4, 5]
        )

    # --------------------------------------------------------
    # Find pixels where any optical channel is NaN
    # --------------------------------------------------------

    nan_mask = np.isnan(
        optical
    ).any(axis=0)

    nan_count = np.sum(
        nan_mask
    )

    print(
        f"\nNaN pixels: {nan_count:,}"
    )

    # --------------------------------------------------------
    # Locate NaN rows and columns
    # --------------------------------------------------------

    nan_rows, nan_cols = np.where(
        nan_mask
    )

    print(
        f"NaN row range: "
        f"{nan_rows.min()} - {nan_rows.max()}"
    )

    print(
        f"NaN col range: "
        f"{nan_cols.min()} - {nan_cols.max()}"
    )

    unique_columns = np.unique(
        nan_cols
    )

    unique_rows = np.unique(
        nan_rows
    )

    print(
        f"Number of NaN columns: "
        f"{len(unique_columns)}"
    )

    print(
        f"NaN columns: "
        f"{unique_columns}"
    )

    print(
        f"Number of NaN rows: "
        f"{len(unique_rows)}"
    )

    # --------------------------------------------------------
    # Compare with valid mask
    # --------------------------------------------------------

    nan_and_valid = np.sum(
        nan_mask &
        (valid_mask == 1)
    )

    nan_and_invalid = np.sum(
        nan_mask &
        (valid_mask == 0)
    )

    print("\nMASK COMPARISON")

    print(
        f"NaN + valid pixels   : "
        f"{nan_and_valid:,}"
    )

    print(
        f"NaN + invalid pixels : "
        f"{nan_and_invalid:,}"
    )

    if nan_and_valid == 0:

        print(
            "✅ All NaN pixels are marked invalid."
        )

    else:

        print(
            "❌ Some NaN pixels are marked VALID."
        )


print("\n" + "=" * 80)
print("KODAGU NaN / VALID MASK CHECK COMPLETE")
print("=" * 80)