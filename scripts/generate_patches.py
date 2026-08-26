import os
import csv
import numpy as np
import rasterio


# ============================================================
# FORESTWATCH - 14-CHANNEL TEMPORAL PATCH GENERATION
#
# Input:
#   Year 1: 7-channel raster
#   Year 2: 7-channel raster
#
# Output:
#   14-channel temporal patches
#
# Channel order:
#
#   Year 1:
#       1. B2
#       2. B3
#       3. B4
#       4. B8
#       5. NDVI
#       6. VV
#       7. VH
#
#   Year 2:
#       8. B2
#       9. B3
#      10. B4
#      11. B8
#      12. NDVI
#      13. VV
#      14. VH
#
# Label:
#   0 = no forest loss
#   1 = forest loss present
#
# Spatial split:
#   A + D = TRAIN
#   C     = VALIDATION
#   B     = TEST
#
# IMPORTANT:
#   This script creates the actual dataset.
# ============================================================


DATA_FOLDER = "data/processed/normalized"
MASK_FOLDER = "data/processed/masks"
LABEL_FOLDER = "data/labels_aligned"

OUTPUT_FOLDER = "data/processed/patches"
REGIONS = [
    "Chikkamagaluru",
    "Kodagu",
    "UttaraKannada"
]

PERIODS = [
    ("2020", "2022", "2020_2022"),
    ("2022", "2024", "2022_2024")
]

PATCH_SIZE = 224
STRIDE = 112


# ============================================================
# SPATIAL BLOCK ASSIGNMENT
# ============================================================

def get_spatial_block(row_index, col_index, n_rows, n_cols):
    """
    Convert the patch position into a normalized spatial block.

    10x10 patch grid -> 5x5 blocks
    6x6 patch grid   -> 3x3 blocks

    Returns:
        TL, TR, BL, BR
    """

    block_rows = int(np.ceil(n_rows / 2))
    block_cols = int(np.ceil(n_cols / 2))

    block_row = row_index // 2
    block_col = col_index // 2

    # Normalize block position
    row_center = (block_row + 0.5) / block_rows
    col_center = (block_col + 0.5) / block_cols

    if row_center < 0.5 and col_center < 0.5:
        return "TL"

    elif row_center < 0.5 and col_center >= 0.5:
        return "TR"

    elif row_center >= 0.5 and col_center < 0.5:
        return "BL"

    else:
        return "BR"


def get_dataset_split(block):
    """
    Final locked spatial split.

    TL + BR -> TRAIN
    TR       -> VALIDATION
    BL       -> TEST
    """

    if block in ["TL", "BR"]:
        return "train"

    elif block == "TR":
        return "validation"

    elif block == "BL":
        return "test"

    else:
        raise ValueError(
            f"Unknown spatial block: {block}"
        )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():

    for split in [
        "train",
        "validation",
        "test"
    ]:

        split_folder = os.path.join(
            OUTPUT_FOLDER,
            split
        )

        os.makedirs(
            split_folder,
            exist_ok=True
        )


# ============================================================
# PROCESS ONE REGION / PERIOD
# ============================================================

def generate_period(
    region,
    year1,
    year2,
    period
):

    image1_path = os.path.join(
        DATA_FOLDER,
        f"{region}_{year1}_7channel.tif"
    )

    image2_path = os.path.join(
        DATA_FOLDER,
        f"{region}_{year2}_7channel.tif"
    )

    label_path = os.path.join(
        LABEL_FOLDER,
        f"{region}_Loss_{period}.tif"
    )

    mask_path = os.path.join(
        MASK_FOLDER,
        f"{region}_valid_mask.tif"
    )

    required_files = [
        image1_path,
        image2_path,
        label_path,
        mask_path
    ]

    for filepath in required_files:

        if not os.path.exists(filepath):

            raise FileNotFoundError(
                f"Missing file: {filepath}"
            )

    print("\n" + "=" * 80)

    print(
        f"{region} | {year1} -> {year2}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Open datasets
    # --------------------------------------------------------

    with rasterio.open(image1_path) as src1, \
         rasterio.open(image2_path) as src2, \
         rasterio.open(label_path) as label_src, \
         rasterio.open(mask_path) as mask_src:

        # ----------------------------------------------------
        # Verify dimensions
        # ----------------------------------------------------

        height = src1.height
        width = src1.width

        if src1.count != 7:
            raise ValueError(
                f"{image1_path} must have 7 bands."
            )

        if src2.count != 7:
            raise ValueError(
                f"{image2_path} must have 7 bands."
            )

        if (
            src2.height != height
            or src2.width != width
        ):
            raise ValueError(
                "Year 1 and Year 2 dimensions do not match."
            )

        if (
            label_src.height != height
            or label_src.width != width
        ):
            raise ValueError(
                "Label dimensions do not match."
            )

        if (
            mask_src.height != height
            or mask_src.width != width
        ):
            raise ValueError(
                "Valid mask dimensions do not match."
            )

        # ----------------------------------------------------
        # Read data
        # ----------------------------------------------------

        image1 = src1.read()

        image2 = src2.read()

        label = label_src.read(1)

        valid_mask = mask_src.read(1)

    # --------------------------------------------------------
    # Create patch coordinate grid
    # --------------------------------------------------------

    row_positions = list(
        range(
            0,
            height - PATCH_SIZE + 1,
            STRIDE
        )
    )

    col_positions = list(
        range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        )
    )

    n_rows = len(row_positions)
    n_cols = len(col_positions)

    print(
        f"Scene size : {width} x {height}"
    )

    print(
        f"Patch grid : {n_rows} x {n_cols}"
    )

    print(
        f"Total patches: "
        f"{n_rows * n_cols}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = {

        "train": {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "discarded": 0
        },

        "validation": {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "discarded": 0
        },

        "test": {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "discarded": 0
        }
    }

    # --------------------------------------------------------
    # Generate patches
    # --------------------------------------------------------

    for row_index, row in enumerate(
        row_positions
    ):

        for col_index, col in enumerate(
            col_positions
        ):

            block = get_spatial_block(
                row_index,
                col_index,
                n_rows,
                n_cols
            )

            split = get_dataset_split(
                block
            )

            statistics[split]["total"] += 1

            # ----------------------------------------------
            # Extract valid mask
            # ----------------------------------------------

            valid_patch = valid_mask[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            valid_pixels = np.sum(
                valid_patch == 1
            )

            # ----------------------------------------------
            # Discard completely invalid patches
            # ----------------------------------------------

            if valid_pixels == 0:

                statistics[split]["discarded"] += 1

                continue

            # ----------------------------------------------
            # Extract label
            # ----------------------------------------------

            label_patch = label[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            # ----------------------------------------------
            # Only count loss pixels inside valid area
            # ----------------------------------------------

            positive_pixels = np.sum(
                (label_patch == 1)
                &
                (valid_patch == 1)
            )

            positive_percentage = (
                positive_pixels /
                valid_pixels
            ) * 100

            # ----------------------------------------------
            # Patch classification
            #
            # Any valid forest-loss pixel:
            #     positive
            #
            # No forest-loss pixel:
            #     negative
            # ----------------------------------------------

            if positive_pixels > 0:

                patch_label = 1

                statistics[split]["positive"] += 1

            else:

                patch_label = 0

                statistics[split]["negative"] += 1

            # ----------------------------------------------
            # Extract 7-channel Year 1 patch
            # ----------------------------------------------

            patch1 = image1[
                :,
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            # ----------------------------------------------
            # Extract 7-channel Year 2 patch
            # ----------------------------------------------

            patch2 = image2[
                :,
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

           # --------------------------------------------------------
            # Combine Year 1 and Year 2 into a 14-channel patch
            # --------------------------------------------------------

            temporal_patch = np.concatenate(
                [patch1, patch2],
                axis=0
            )

            # --------------------------------------------------------
            # Handle NaN values in invalid pixels
            #
            # NaNs occur only where valid_mask == 0.
            # These pixels are excluded from label calculation.
            # --------------------------------------------------------

            invalid_patch = (valid_patch == 0)

            nan_mask = np.isnan(temporal_patch)

            replace_mask = (
                nan_mask &
                invalid_patch[np.newaxis, :, :]
            )

            temporal_patch[replace_mask] = 0.0

            # Safety check: no NaNs should remain
            if np.isnan(temporal_patch).any():
                raise ValueError(
                    "NaN values remain in patch after "
                    "invalid-pixel handling."
                )

            # --------------------------------------------------------
            # Verify patch shape
            # --------------------------------------------------------

            expected_shape = (
                14,
                PATCH_SIZE,
                PATCH_SIZE
            )

            if temporal_patch.shape != expected_shape:
                raise ValueError(
                    f"Unexpected patch shape: "
                    f"{temporal_patch.shape}"
                )
            # ----------------------------------------------
            # Verify shape
            # ----------------------------------------------

            expected_shape = (
                14,
                PATCH_SIZE,
                PATCH_SIZE
            )

            if temporal_patch.shape != expected_shape:

                raise ValueError(
                    f"Unexpected patch shape: "
                    f"{temporal_patch.shape}"
                )

            # ----------------------------------------------
            # Output filename
            # ----------------------------------------------

            filename = (
                f"{region}_"
                f"{period}_"
                f"r{row}_"
                f"c{col}_"
                f"label{patch_label}.npy"
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                split,
                filename
            )

            # ----------------------------------------------
            # Save patch
            # ----------------------------------------------

            np.save(
                output_path,
                temporal_patch.astype(
                    np.float32
                )
            )

            # ----------------------------------------------
            # Save metadata
            # ----------------------------------------------

            metadata_path = os.path.join(
                OUTPUT_FOLDER,
                "patch_metadata.csv"
            )

            metadata_exists = os.path.exists(
                metadata_path
            )

            with open(
                metadata_path,
                "a",
                newline=""
            ) as metadata_file:

                writer = csv.writer(
                    metadata_file
                )

                if not metadata_exists:

                    writer.writerow(
                        [
                            "region",
                            "period",
                            "year1",
                            "year2",
                            "row",
                            "column",
                            "grid_row",
                            "grid_column",
                            "spatial_block",
                            "split",
                            "label",
                            "positive_pixels",
                            "valid_pixels",
                            "positive_percentage",
                            "filename"
                        ]
                    )

                writer.writerow(
                    [
                        region,
                        period,
                        year1,
                        year2,
                        row,
                        col,
                        row_index + 1,
                        col_index + 1,
                        block,
                        split,
                        patch_label,
                        int(positive_pixels),
                        int(valid_pixels),
                        float(
                            positive_percentage
                        ),
                        filename
                    ]
                )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print("\nPATCH GENERATION SUMMARY")

    for split in [
        "train",
        "validation",
        "test"
    ]:

        stats = statistics[split]

        print(
            f"\n{split.upper()}"
        )

        print(
            f"  Total grid patches : "
            f"{stats['total']}"
        )

        print(
            f"  Positive patches   : "
            f"{stats['positive']}"
        )

        print(
            f"  Negative patches   : "
            f"{stats['negative']}"
        )

        print(
            f"  Discarded patches  : "
            f"{stats['discarded']}"
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)

print(
    "FORESTWATCH - 14-CHANNEL TEMPORAL PATCH GENERATION"
)

print("=" * 80)

print(
    "\nInput:"
)

print(
    "  Year 1 = 7 channels"
)

print(
    "  Year 2 = 7 channels"
)

print(
    "  Combined = 14 channels"
)

print(
    "\nPatch size:"
)

print(
    f"  {PATCH_SIZE} x {PATCH_SIZE}"
)

print(
    "\nStride:"
)

print(
    f"  {STRIDE}"
)

print(
    "\nSpatial split:"
)

print(
    "  A + D -> TRAIN"
)

print(
    "  C     -> VALIDATION"
)

print(
    "  B     -> TEST"
)

print(
    "\nLabel:"
)

print(
    "  0 = no forest loss"
)

print(
    "  1 = forest loss present"
)

print(
    "\nCreating output directories..."
)

create_output_directories()


for region in REGIONS:

    for year1, year2, period in PERIODS:

        generate_period(
            region,
            year1,
            year2,
            period
        )


print("\n" + "=" * 80)

print(
    "14-CHANNEL PATCH GENERATION COMPLETE"
)

print("=" * 80)

print(
    "\nOutput folder:"
)

print(
    f"  {OUTPUT_FOLDER}"
)

print(
    "\nMetadata:"
)

print(
    f"  {OUTPUT_FOLDER}/patch_metadata.csv"
)