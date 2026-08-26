import os
import numpy as np
import rasterio


# ============================================================
# FORESTWATCH - SPATIAL BLOCK ANALYSIS
#
# Purpose:
#   Analyze class distribution inside spatial blocks before
#   creating the final train / validation / test split.
#
# IMPORTANT:
#   This script DOES NOT create training patches.
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

    print("\n" + "=" * 80)
    print(f"{region} | {year1} -> {year2}")
    print("=" * 80)

    print(
        f"Scene      : {width} x {height}"
    )

    print(
        f"Patch grid : {n_rows} x {n_cols}"
    )

    # --------------------------------------------------------
    # 2 x 2 block boundaries
    # --------------------------------------------------------

    row_split = n_rows / 2
    col_split = n_cols / 2

    blocks = {
        "A": [],
        "B": [],
        "C": [],
        "D": []
    }

    # --------------------------------------------------------
    # Assign patches to blocks
    #
    # A = top-left
    # B = top-right
    # C = bottom-left
    # D = bottom-right
    # --------------------------------------------------------

    for row_index, row in enumerate(row_positions):

        for col_index, col in enumerate(col_positions):

            if row_index < row_split:

                if col_index < col_split:
                    block = "A"
                else:
                    block = "B"

            else:

                if col_index < col_split:
                    block = "C"
                else:
                    block = "D"

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
                continue

            positive_pixels = np.sum(
                (label_patch == 1) &
                (valid_patch == 1)
            )

            positive_percentage = (
                positive_pixels /
                valid_pixels
            ) * 100

            if positive_pixels == 0:
                patch_class = "negative"
            else:
                patch_class = "positive"

            blocks[block].append(
                {
                    "row": row_index + 1,
                    "col": col_index + 1,
                    "positive_pixels": positive_pixels,
                    "positive_percentage": positive_percentage,
                    "class": patch_class
                }
            )

    # --------------------------------------------------------
    # Report each block
    # --------------------------------------------------------

    print("\nBLOCK DISTRIBUTION")
    print("-" * 80)

    for block_name in ["A", "B", "C", "D"]:

        patches = blocks[block_name]

        total = len(patches)

        positive = sum(
            p["class"] == "positive"
            for p in patches
        )

        negative = sum(
            p["class"] == "negative"
            for p in patches
        )

        positive_pixels = sum(
            p["positive_pixels"]
            for p in patches
        )

        print(
            f"\nBlock {block_name}"
        )

        print(
            f"  Total patches    : {total}"
        )

        print(
            f"  Positive patches : {positive}"
        )

        print(
            f"  Negative patches : {negative}"
        )

        if total > 0:

            print(
                f"  Positive ratio   : "
                f"{positive / total * 100:.2f}%"
            )

        print(
            f"  Positive pixels  : "
            f"{positive_pixels:,}"
        )

    # --------------------------------------------------------
    # Visual block map
    # --------------------------------------------------------

    print("\nBLOCK MAP")

    print(
        "A = top-left | B = top-right"
    )

    print(
        "C = bottom-left | D = bottom-right"
    )

    print()

    for row_index in range(n_rows):

        for col_index in range(n_cols):

            if row_index < row_split:

                if col_index < col_split:
                    block = "A"
                else:
                    block = "B"

            else:

                if col_index < col_split:
                    block = "C"
                else:
                    block = "D"

            print(
                f"{block} ",
                end=""
            )

        print()


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("FORESTWATCH - SPATIAL BLOCK ANALYSIS")
print("=" * 80)

print(
    "\nPatch size : 224 x 224"
)

print(
    "Stride     : 112"
)

print(
    "\n2 x 2 spatial blocks:"
)

print(
    "  A = top-left"
)

print(
    "  B = top-right"
)

print(
    "  C = bottom-left"
)

print(
    "  D = bottom-right"
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
print("SPATIAL BLOCK ANALYSIS COMPLETE")
print("=" * 80)