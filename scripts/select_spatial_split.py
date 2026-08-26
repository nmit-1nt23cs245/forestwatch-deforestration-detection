import os
import itertools
import numpy as np
import rasterio


# ============================================================
# FORESTWATCH - AUTOMATIC SPATIAL SPLIT SELECTION
#
# Searches spatial block assignments and selects a split that:
#   1. Keeps spatial blocks completely separated.
#   2. Gives validation/test positive samples where possible.
#   3. Keeps the combined class distribution reasonable.
#   4. Uses approximately 60/20/20 spatial-block allocation.
#
# IMPORTANT:
#   Analysis only.
#   No patches are created.
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
# LOAD PATCH INFORMATION
# ============================================================

def load_dataset(region, period):

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

    n_rows = len(rows)
    n_cols = len(cols)

    blocks = {}

    for r, row in enumerate(rows):

        for c, col in enumerate(cols):

            valid_patch = valid_mask[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            valid_pixels = np.sum(
                valid_patch == 1
            )

            if valid_pixels == 0:
                continue

            label_patch = label[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            positive_pixels = np.sum(
                (label_patch == 1)
                &
                (valid_patch == 1)
            )

            patch_class = (
                1 if positive_pixels > 0
                else 0
            )

            # Each block contains 2 x 2 patches.
            block = (
                r // 2,
                c // 2
            )

            if block not in blocks:

                blocks[block] = {
                    "total": 0,
                    "positive": 0,
                    "negative": 0
                }

            blocks[block]["total"] += 1

            if patch_class == 1:
                blocks[block]["positive"] += 1
            else:
                blocks[block]["negative"] += 1

    return n_rows, n_cols, blocks


# ============================================================
# COLLECT ALL DATA
# ============================================================

def collect_data():

    datasets = {}

    for region in REGIONS:

        for _, _, period in PERIODS:

            key = (
                region,
                period
            )

            datasets[key] = load_dataset(
                region,
                period
            )

    return datasets


# ============================================================
# SCORE ONE SPLIT
# ============================================================

def score_split(
    datasets,
    train_blocks,
    validation_blocks,
    test_blocks
):

    totals = {
        "train": {
            "total": 0,
            "positive": 0,
            "negative": 0
        },
        "validation": {
            "total": 0,
            "positive": 0,
            "negative": 0
        },
        "test": {
            "total": 0,
            "positive": 0,
            "negative": 0
        }
    }

    # --------------------------------------------------------
    # Apply same relative block assignment to each dataset.
    #
    # For 10x10 scenes:
    #   5x5 blocks
    #
    # For 6x6 scenes:
    #   3x3 blocks
    #
    # We normalize block coordinates to [0,1] so that the
    # assignment is spatially relative rather than dependent
    # on the absolute number of blocks.
    # --------------------------------------------------------

    for (
        region,
        period
    ), (
        n_rows,
        n_cols,
        blocks
    ) in datasets.items():

        max_block_row = max(
            b[0] for b in blocks
        )

        max_block_col = max(
            b[1] for b in blocks
        )

        for block, stats in blocks.items():

            br, bc = block

            # Normalized block-center coordinates
            r_norm = (
                br + 0.5
            ) / (
                max_block_row + 1
            )

            c_norm = (
                bc + 0.5
            ) / (
                max_block_col + 1
            )

            # Find closest canonical spatial position.
            #
            # This lets the same spatial regions be assigned
            # consistently despite different grid sizes.

            if (
                r_norm < 1 / 3
                and c_norm < 1 / 3
            ):
                canonical = "TL"

            elif (
                r_norm < 1 / 3
                and c_norm >= 2 / 3
            ):
                canonical = "TR"

            elif (
                r_norm >= 2 / 3
                and c_norm < 1 / 3
            ):
                canonical = "BL"

            elif (
                r_norm >= 2 / 3
                and c_norm >= 2 / 3
            ):
                canonical = "BR"

            elif r_norm < 0.5:
                canonical = "TOP"

            elif r_norm >= 0.5:
                canonical = "BOTTOM"

            elif c_norm < 0.5:
                canonical = "LEFT"

            else:
                canonical = "RIGHT"

            # ------------------------------------------------
            # Determine split from supplied canonical groups.
            # ------------------------------------------------

            if canonical in train_blocks:
                split = "train"

            elif canonical in validation_blocks:
                split = "validation"

            elif canonical in test_blocks:
                split = "test"

            else:
                # Middle blocks are assigned using nearest
                # canonical region.
                #
                # This fallback prevents any block from being
                # silently discarded.
                split = "train"

            totals[split]["total"] += stats["total"]

            totals[split]["positive"] += stats["positive"]

            totals[split]["negative"] += stats["negative"]

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    total_samples = sum(
        totals[s]["total"]
        for s in totals
    )

    total_positive = sum(
        totals[s]["positive"]
        for s in totals
    )

    # Class ratio difference
    global_positive_ratio = (
        total_positive / total_samples
        if total_samples > 0
        else 0
    )

    score = 0.0

    # --------------------------------------------------------
    # Desired split proportions
    # --------------------------------------------------------

    desired = {
        "train": 0.60,
        "validation": 0.20,
        "test": 0.20
    }

    for split in totals:

        actual_ratio = (
            totals[split]["total"] /
            total_samples
        )

        score -= abs(
            actual_ratio -
            desired[split]
        ) * 100

    # --------------------------------------------------------
    # Class-balance objective
    # --------------------------------------------------------

    for split in totals:

        total = totals[split]["total"]
        positive = totals[split]["positive"]

        if total == 0:
            score -= 1000
            continue

        ratio = positive / total

        score -= abs(
            ratio -
            global_positive_ratio
        ) * 100

        # Penalize completely single-class validation/test
        if split in [
            "validation",
            "test"
        ]:

            if positive == 0:
                score -= 100

            if positive == total:
                score -= 100

    return score, totals


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FORESTWATCH - AUTOMATIC SPATIAL SPLIT SELECTION")
    print("=" * 80)

    print(
        "\nBuilding spatial-block statistics..."
    )

    datasets = collect_data()

    print(
        "\nSearching candidate spatial assignments..."
    )

    # --------------------------------------------------------
    # Canonical spatial groups
    # --------------------------------------------------------
    #
    # We intentionally keep the number of candidate
    # assignments small and interpretable.
    #
    # The final split will remain spatially meaningful.
    # --------------------------------------------------------

    candidates = [

        (
            {"TL", "TR", "TOP"},
            {"BL"},
            {"BR"}
        ),

        (
            {"TL", "BL", "LEFT"},
            {"TR"},
            {"BR"}
        ),

        (
            {"TL", "BR"},
            {"TR"},
            {"BL"}
        ),

        (
            {"TL", "BR", "BOTTOM"},
            {"TR"},
            {"BL"}
        ),

        (
            {"TL", "TR", "BL"},
            {"BR"},
            {"BOTTOM"}
        ),

        (
            {"TL", "TR", "BL", "LEFT"},
            {"BR"},
            {"RIGHT"}
        )
    ]

    results = []

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        train_blocks = candidate[0]
        validation_blocks = candidate[1]
        test_blocks = candidate[2]

        score, totals = score_split(
            datasets,
            train_blocks,
            validation_blocks,
            test_blocks
        )

        results.append(
            (
                score,
                index,
                train_blocks,
                validation_blocks,
                test_blocks,
                totals
            )
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # Print top candidates
    # --------------------------------------------------------

    print(
        "\nTOP CANDIDATE SPLITS"
    )

    print("-" * 80)

    for rank, result in enumerate(
        results[:5],
        start=1
    ):

        (
            score,
            index,
            train_blocks,
            validation_blocks,
            test_blocks,
            totals
        ) = result

        print(
            f"\nRANK {rank}"
        )

        print(
            f"  Score: {score:.2f}"
        )

        print(
            f"  Train blocks      : "
            f"{sorted(train_blocks)}"
        )

        print(
            f"  Validation blocks : "
            f"{sorted(validation_blocks)}"
        )

        print(
            f"  Test blocks       : "
            f"{sorted(test_blocks)}"
        )

        for split in [
            "train",
            "validation",
            "test"
        ]:

            stats = totals[split]

            ratio = (
                stats["positive"] /
                stats["total"] * 100
                if stats["total"] > 0
                else 0
            )

            print(
                f"  {split.upper():11s}: "
                f"total={stats['total']:3d}, "
                f"positive={stats['positive']:3d}, "
                f"negative={stats['negative']:3d}, "
                f"positive_ratio={ratio:6.2f}%"
            )

    # --------------------------------------------------------
    # Best candidate
    # --------------------------------------------------------

    best = results[0]

    (
        score,
        index,
        train_blocks,
        validation_blocks,
        test_blocks,
        totals
    ) = best

    print("\n" + "=" * 80)

    print(
        "RECOMMENDED CANDIDATE"
    )

    print("=" * 80)

    print(
        f"\nCandidate : {index}"
    )

    print(
        f"Score     : {score:.2f}"
    )

    print(
        f"\nTrain blocks:"
    )

    print(
        f"  {sorted(train_blocks)}"
    )

    print(
        f"\nValidation blocks:"
    )

    print(
        f"  {sorted(validation_blocks)}"
    )

    print(
        f"\nTest blocks:"
    )

    print(
        f"  {sorted(test_blocks)}"
    )

    print(
        "\nClass distribution:"
    )

    for split in [
        "train",
        "validation",
        "test"
    ]:

        stats = totals[split]

        ratio = (
            stats["positive"] /
            stats["total"] * 100
            if stats["total"] > 0
            else 0
        )

        print(
            f"  {split.upper():11s}: "
            f"{stats['total']} total | "
            f"{stats['positive']} positive | "
            f"{stats['negative']} negative | "
            f"{ratio:.2f}% positive"
        )

    print(
        "\nNOTE:"
    )

    print(
        "This is a candidate recommendation only."
    )

    print(
        "We will verify the selected split before "
        "regenerating patches."
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()