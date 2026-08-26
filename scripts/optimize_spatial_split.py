import os
import numpy as np
import rasterio


# ============================================================
# FORESTWATCH - SPATIAL SPLIT OPTIMIZATION
#
# Purpose:
#   Test smaller spatial blocks and identify a defensible
#   train / validation / test split.
#
# IMPORTANT:
#   Analysis only.
#   No training patches are created.
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
# READ PATCH CLASS INFORMATION
# ============================================================

def get_patch_grid(region, period):

    label_path = os.path.join(
        LABEL_FOLDER,
        f"{region}_Loss_{period}.tif"
    )

    mask_path = os.path.join(
        MASK_FOLDER,
        f"{region}_valid_mask.tif"
    )

    with rasterio.open(label_path) as label_src, \
         rasterio.open(mask_path) as mask_src:

        height = label_src.height
        width = label_src.width

        label = label_src.read(1)
        valid_mask = mask_src.read(1)

    rows = list(
        range(
            0,
            height - PATCH_SIZE + 1,
            STRIDE
        )
    )

    cols = list(
        range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        )
    )

    grid = []

    for r, row in enumerate(rows):

        for c, col in enumerate(cols):

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
                patch_class = None

            else:

                positive_pixels = np.sum(
                    (label_patch == 1)
                    &
                    (valid_patch == 1)
                )

                if positive_pixels > 0:
                    patch_class = 1
                else:
                    patch_class = 0

            grid.append(
                {
                    "row": r,
                    "col": c,
                    "class": patch_class
                }
            )

    return len(rows), len(cols), grid


# ============================================================
# CREATE SMALL BLOCKS
# ============================================================

def assign_small_block(
    row,
    col,
    n_rows,
    n_cols
):

    # --------------------------------------------------------
    # Divide each dimension into approximately two-patch
    # blocks.
    #
    # 10x10 -> 5x5 blocks
    #  6x6  -> 3x3 blocks
    # --------------------------------------------------------

    block_rows = int(
        np.ceil(n_rows / 2)
    )

    block_cols = int(
        np.ceil(n_cols / 2)
    )

    block_row = row // 2
    block_col = col // 2

    return (
        block_row,
        block_col
    )


# ============================================================
# ANALYZE ONE DATASET
# ============================================================

def analyze_dataset(
    region,
    year1,
    year2,
    period
):

    n_rows, n_cols, grid = get_patch_grid(
        region,
        period
    )

    print("\n" + "=" * 80)

    print(
        f"{region} | {year1} -> {year2}"
    )

    print("=" * 80)

    print(
        f"Patch grid : {n_rows} x {n_cols}"
    )

    blocks = {}

    for patch in grid:

        block = assign_small_block(
            patch["row"],
            patch["col"],
            n_rows,
            n_cols
        )

        if block not in blocks:

            blocks[block] = {
                "total": 0,
                "positive": 0,
                "negative": 0
            }

        if patch["class"] is None:
            continue

        blocks[block]["total"] += 1

        if patch["class"] == 1:
            blocks[block]["positive"] += 1

        else:
            blocks[block]["negative"] += 1

    print(
        "\nSMALL SPATIAL BLOCKS"
    )

    print("-" * 80)

    for block in sorted(blocks):

        stats = blocks[block]

        total = stats["total"]
        positive = stats["positive"]
        negative = stats["negative"]

        ratio = (
            positive / total * 100
            if total > 0
            else 0
        )

        print(
            f"Block {block}: "
            f"total={total:2d}, "
            f"positive={positive:2d}, "
            f"negative={negative:2d}, "
            f"positive_ratio={ratio:6.2f}%"
        )

    return blocks


# ============================================================
# COMBINED BLOCK ANALYSIS
# ============================================================

def main():

    print("=" * 80)
    print("FORESTWATCH - SMALL SPATIAL BLOCK ANALYSIS")
    print("=" * 80)

    all_results = {}

    for region in REGIONS:

        for year1, year2, period in PERIODS:

            key = (
                region,
                period
            )

            all_results[key] = analyze_dataset(
                region,
                year1,
                year2,
                period
            )

    # --------------------------------------------------------
    # Combined statistics by spatial block
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("COMBINED SPATIAL BLOCK STATISTICS")
    print("=" * 80)

    combined = {}

    for dataset_blocks in all_results.values():

        for block, stats in dataset_blocks.items():

            if block not in combined:

                combined[block] = {
                    "total": 0,
                    "positive": 0,
                    "negative": 0
                }

            combined[block]["total"] += (
                stats["total"]
            )

            combined[block]["positive"] += (
                stats["positive"]
            )

            combined[block]["negative"] += (
                stats["negative"]
            )

    print()

    for block in sorted(combined):

        stats = combined[block]

        total = stats["total"]
        positive = stats["positive"]
        negative = stats["negative"]

        ratio = (
            positive / total * 100
            if total > 0
            else 0
        )

        print(
            f"Block {block}: "
            f"total={total:3d}, "
            f"positive={positive:3d}, "
            f"negative={negative:3d}, "
            f"positive_ratio={ratio:6.2f}%"
        )

    print("\n" + "=" * 80)
    print("SMALL SPATIAL BLOCK ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()