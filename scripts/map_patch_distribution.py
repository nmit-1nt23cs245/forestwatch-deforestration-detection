import os
import numpy as np
import rasterio


# ============================================================
# FORESTWATCH - PATCH DISTRIBUTION MAP
#
# Purpose:
#   Show the spatial location and class of every possible
#   224 x 224 patch.
#
# This script DOES NOT create training patches.
# ============================================================


LABEL_FOLDER = "data/labels_aligned"
MASK_FOLDER = "data/processed/masks"

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
# ANALYZE ONE REGION / PERIOD
# ============================================================

def analyze_period(region, year1, year2, period):

    label_path = os.path.join(
        LABEL_FOLDER,
        f"{region}_Loss_{period}.tif"
    )

    mask_path = os.path.join(
        MASK_FOLDER,
        f"{region}_valid_mask.tif"
    )

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Missing label: {label_path}"
        )

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Missing mask: {mask_path}"
        )

    with rasterio.open(label_path) as label_src, \
         rasterio.open(mask_path) as mask_src:

        height = label_src.height
        width = label_src.width

        label = label_src.read(1)
        valid_mask = mask_src.read(1)

    print("\n" + "=" * 80)
    print(
        f"{region} | {year1} -> {year2}"
    )
    print("=" * 80)

    print(
        f"Scene size: {width} x {height}"
    )

    print(
        f"Patch grid: "
        f"{len(range(0, height - PATCH_SIZE + 1, STRIDE))} rows x "
        f"{len(range(0, width - PATCH_SIZE + 1, STRIDE))} columns"
    )

    print("\nPATCH MAP")
    print(
        "Each cell represents one 224x224 patch."
    )
    print(
        "N = negative (0% loss)"
    )
    print(
        "P = positive (any loss)"
    )

    positive_locations = []

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

    # --------------------------------------------------------
    # Print column header
    # --------------------------------------------------------

    print("\n     ", end="")

    for col_index in range(len(col_positions)):
        print(
            f"{col_index + 1:3}",
            end=""
        )

    print()

    print(
        "     " +
        "---" * len(col_positions)
    )

    # --------------------------------------------------------
    # Analyze each patch
    # --------------------------------------------------------

    for row_index, row in enumerate(row_positions):

        print(
            f"{row_index + 1:3} |",
            end=""
        )

        for col_index, col in enumerate(col_positions):

            label_patch = label[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            valid_patch = valid_mask[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            valid_pixels = np.sum(
                valid_patch == 1
            )

            if valid_pixels == 0:

                symbol = "X"

                positive_percentage = 0.0

            else:

                positive_pixels = np.sum(
                    (label_patch == 1) &
                    (valid_patch == 1)
                )

                positive_percentage = (
                    positive_pixels /
                    valid_pixels
                ) * 100

                if positive_pixels > 0:
                    symbol = "P"
                else:
                    symbol = "N"

            print(
                f"{symbol:3}",
                end=""
            )

            if symbol == "P":

                positive_locations.append(
                    (
                        row_index + 1,
                        col_index + 1,
                        row,
                        col,
                        positive_percentage
                    )
                )

        print()

    # --------------------------------------------------------
    # Positive patch details
    # --------------------------------------------------------

    print(
        "\nPOSITIVE PATCH LOCATIONS"
    )

    print(
        "-" * 80
    )

    if not positive_locations:

        print(
            "No positive patches found."
        )

    else:

        print(
            f"{'Grid':<10}"
            f"{'Row':<8}"
            f"{'Col':<8}"
            f"{'Loss %':>12}"
        )

        print(
            "-" * 40
        )

        for (
            grid_row,
            grid_col,
            pixel_row,
            pixel_col,
            positive_percentage
        ) in positive_locations:

            print(
                f"({grid_row},{grid_col})"
                f"{pixel_row:<8}"
                f"{pixel_col:<8}"
                f"{positive_percentage:>11.6f}%"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_patches = (
        len(row_positions) *
        len(col_positions)
    )

    positive_count = len(
        positive_locations
    )

    negative_count = (
        total_patches -
        positive_count
    )

    print(
        "\nSUMMARY"
    )

    print(
        f"Total patches    : {total_patches}"
    )

    print(
        f"Positive patches : {positive_count}"
    )

    print(
        f"Negative patches : {negative_count}"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("FORESTWATCH - PATCH DISTRIBUTION MAP")
print("=" * 80)

print(
    "\nPatch size : 224 x 224"
)

print(
    "Stride     : 112"
)

print(
    "\nNOTE:"
)

print(
    "This is analysis only."
)

print(
    "No training patches are being created."
)


for region in REGIONS:

    for year1, year2, period in PERIODS:

        analyze_period(
            region,
            year1,
            year2,
            period
        )


print("\n" + "=" * 80)
print("PATCH DISTRIBUTION MAP COMPLETE")
print("=" * 80)