import os
import rasterio


# ============================================================
# FORESTWATCH - SPATIAL TRAIN / VALIDATION / TEST SPLIT ANALYSIS
#
# Purpose:
#   Design and verify a spatially separated patch split.
#
# IMPORTANT:
#   This script does NOT create training patches.
#   It only analyzes the patch coordinates and split.
# ============================================================


DATA_FOLDER = "data/processed/normalized"

REGIONS = [
    "Chikkamagaluru",
    "Kodagu",
    "UttaraKannada"
]

YEARS = [
    "2020",
    "2022",
    "2024"
]

PATCH_SIZE = 224
STRIDE = 112


# ============================================================
# SPATIAL SPLIT
# ============================================================
#
# We divide the image into horizontal spatial zones:
#
#   Top 70%       -> TRAIN
#   Next 15%      -> VALIDATION
#   Bottom 15%    -> TEST
#
# The assignment is based on the patch center row.
#
# This prevents neighboring overlapping patches from being
# randomly distributed between train/validation/test.
# ============================================================


def get_split(row, height):

    patch_center_row = row + (PATCH_SIZE // 2)

    train_limit = height * 0.70
    validation_limit = height * 0.85

    if patch_center_row < train_limit:
        return "train"

    elif patch_center_row < validation_limit:
        return "validation"

    else:
        return "test"


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(region, year):

    image_path = os.path.join(
        DATA_FOLDER,
        f"{region}_{year}_7channel.tif"
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Missing file: {image_path}"
        )

    with rasterio.open(image_path) as src:

        height = src.height
        width = src.width

    print("\n" + "-" * 80)
    print(f"{region} | {year}")
    print("-" * 80)

    print(f"Scene size : {width} x {height}")
    print(f"Patch size : {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Stride     : {STRIDE}")

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    split_rows = {
        "train": [],
        "validation": [],
        "test": []
    }

    split_columns = {
        "train": [],
        "validation": [],
        "test": []
    }

    # --------------------------------------------------------
    # Generate patch coordinates
    # --------------------------------------------------------

    for row in range(
        0,
        height - PATCH_SIZE + 1,
        STRIDE
    ):

        for col in range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        ):

            split = get_split(
                row,
                height
            )

            split_counts[split] += 1

            split_rows[split].append(row)
            split_columns[split].append(col)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    total_patches = sum(
        split_counts.values()
    )

    print(
        f"\nTotal patches : {total_patches}"
    )

    for split in [
        "train",
        "validation",
        "test"
    ]:

        count = split_counts[split]

        percentage = (
            count / total_patches * 100
            if total_patches > 0
            else 0
        )

        print(
            f"\n{split.upper()}"
        )

        print(
            f"  Patches     : {count}"
        )

        print(
            f"  Percentage  : {percentage:.2f}%"
        )

        if split_rows[split]:

            print(
                f"  Row range   : "
                f"{min(split_rows[split])} - "
                f"{max(split_rows[split])}"
            )

            print(
                f"  Column range: "
                f"{min(split_columns[split])} - "
                f"{max(split_columns[split])}"
            )

    # --------------------------------------------------------
    # Check split uniqueness
    # --------------------------------------------------------

    train_coords = set(
        zip(
            split_rows["train"],
            split_columns["train"]
        )
    )

    validation_coords = set(
        zip(
            split_rows["validation"],
            split_columns["validation"]
        )
    )

    test_coords = set(
        zip(
            split_rows["test"],
            split_columns["test"]
        )
    )

    train_validation_overlap = (
        train_coords & validation_coords
    )

    train_test_overlap = (
        train_coords & test_coords
    )

    validation_test_overlap = (
        validation_coords & test_coords
    )

    print("\nSPLIT OVERLAP CHECK")

    print(
        f"  Train ∩ Validation : "
        f"{len(train_validation_overlap)}"
    )

    print(
        f"  Train ∩ Test       : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"  Validation ∩ Test  : "
        f"{len(validation_test_overlap)}"
    )

    if (
        len(train_validation_overlap) == 0
        and
        len(train_test_overlap) == 0
        and
        len(validation_test_overlap) == 0
    ):

        print(
            "\n  ✅ No patch-coordinate overlap detected."
        )

    else:

        print(
            "\n  ❌ OVERLAP DETECTED!"
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("FORESTWATCH - SPATIAL SPLIT ANALYSIS")
print("=" * 80)

print(
    f"\nPatch size : {PATCH_SIZE} x {PATCH_SIZE}"
)

print(
    f"Stride     : {STRIDE}"
)

print(
    "\nSplit strategy:"
)

print(
    "  Top 70%    -> TRAIN"
)

print(
    "  Next 15%   -> VALIDATION"
)

print(
    "  Bottom 15% -> TEST"
)

print(
    "\nNOTE:"
)

print(
    "This script only analyzes the spatial split."
)

print(
    "No training patches are being created."
)


for region in REGIONS:

    for year in YEARS:

        analyze_image(
            region,
            year
        )


print("\n" + "=" * 80)
print("SPATIAL SPLIT ANALYSIS COMPLETE")
print("=" * 80)