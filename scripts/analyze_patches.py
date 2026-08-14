import os
import numpy as np
import rasterio


# ============================================================
# FORESTWATCH - PATCH DISTRIBUTION ANALYSIS
#
# This script DOES NOT create training patches.
#
# It analyzes possible 224x224 patches for:
#
#   2020 -> 2022
#   2022 -> 2024
#
# and reports:
#
#   - total patches
#   - valid-data percentage
#   - positive/forest-loss pixels
#   - patches containing forest loss
#   - patches containing no forest loss
#
# This helps us design the final sampling strategy.
# ============================================================


DATA_FOLDER = "data/processed/normalized"
MASK_FOLDER = "data/processed/masks"
LABEL_FOLDER = "data/labels_aligned"


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

# 50% overlap.
# We use this only for analysis at this stage.
STRIDE = 112


# ============================================================
# PATCH ANALYSIS FUNCTION
# ============================================================

def analyze_period(
    region,
    year1,
    year2,
    period,
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


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Open files
    # --------------------------------------------------------

    with rasterio.open(image1_path) as src1, \
         rasterio.open(image2_path) as src2, \
         rasterio.open(label_path) as label_src, \
         rasterio.open(mask_path) as mask_src:

        height = src1.height
        width = src1.width

        # ----------------------------------------------------
        # Verify all dimensions
        # ----------------------------------------------------

        if (
            src2.height != height
            or src2.width != width
        ):
            raise ValueError(
                f"Year dimensions do not match for "
                f"{region} {year1}-{year2}"
            )

        if (
            label_src.height != height
            or label_src.width != width
        ):
            raise ValueError(
                f"Label dimensions do not match for "
                f"{region} {period}"
            )

        if (
            mask_src.height != height
            or mask_src.width != width
        ):
            raise ValueError(
                f"Mask dimensions do not match for "
                f"{region}"
            )


        # ----------------------------------------------------
        # Read label and validity mask
        #
        # We don't need to load all 14 feature channels
        # for this analysis.
        # ----------------------------------------------------

        label = label_src.read(1)

        valid_mask = mask_src.read(1)


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_patches = 0

        valid_patches = 0

        rejected_patches = 0

        positive_patches = 0

        negative_patches = 0

        mixed_patches = 0

        positive_pixels_total = 0

        valid_pixels_total = 0

        positive_pixel_percentages = []


        # ----------------------------------------------------
        # Analyze patches
        # ----------------------------------------------------

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

                total_patches += 1


                # --------------------------------------------
                # Extract patch
                # --------------------------------------------

                label_patch = label[
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]

                valid_patch = valid_mask[
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]


                # --------------------------------------------
                # Valid-data statistics
                # --------------------------------------------

                total_pixels = (
                    PATCH_SIZE * PATCH_SIZE
                )

                valid_pixels = np.sum(
                    valid_patch == 1
                )

                valid_percentage = (
                    valid_pixels /
                    total_pixels
                ) * 100


                # --------------------------------------------
                # For analysis, keep ALL patches.
                #
                # We are NOT deciding the final validity
                # threshold yet.
                # --------------------------------------------

                if valid_pixels == 0:

                    rejected_patches += 1

                    continue


                valid_patches += 1


                # --------------------------------------------
                # Forest-loss pixels
                # --------------------------------------------

                positive_pixels = np.sum(
                    label_patch == 1
                )

                positive_pixels_total += (
                    positive_pixels
                )

                valid_pixels_total += (
                    valid_pixels
                )


                if positive_pixels == 0:

                    negative_patches += 1

                elif positive_pixels == total_pixels:

                    positive_patches += 1

                else:

                    mixed_patches += 1


                positive_percentage = (
                    positive_pixels /
                    total_pixels
                ) * 100

                positive_pixel_percentages.append(
                    positive_percentage
                )


    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "-" * 80)

    print(
        f"{region} | "
        f"{year1} -> {year2}"
    )

    print("-" * 80)

    print(
        f"Scene size       : "
        f"{width} x {height}"
    )

    print(
        f"Patch size       : "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Stride           : "
        f"{STRIDE}"
    )

    print(
        f"Total patches    : "
        f"{total_patches:,}"
    )

    print(
        f"Non-empty patches: "
        f"{valid_patches:,}"
    )

    print(
        f"Empty patches    : "
        f"{rejected_patches:,}"
    )

    print(
        f"Positive patches : "
        f"{positive_patches:,}"
    )

    print(
        f"Negative patches : "
        f"{negative_patches:,}"
    )

    print(
        f"Mixed patches    : "
        f"{mixed_patches:,}"
    )

    if valid_patches > 0:

        positive_patch_percentage = (
            (
                positive_patches
                + mixed_patches
            )
            / valid_patches
        ) * 100

        print(
            f"Positive-containing patches: "
            f"{positive_patch_percentage:.2f}%"
        )


    print(
        f"Total positive pixels: "
        f"{positive_pixels_total:,}"
    )

    print(
        f"Total valid pixels: "
        f"{valid_pixels_total:,}"
    )


    if valid_pixels_total > 0:

        positive_area_percentage = (
            positive_pixels_total /
            valid_pixels_total
        ) * 100

        print(
            f"Positive pixel percentage: "
            f"{positive_area_percentage:.6f}%"
        )


    # --------------------------------------------------------
    # Distribution of positive-pixel percentage
    # --------------------------------------------------------

    if positive_pixel_percentages:

        values = np.array(
            positive_pixel_percentages
        )

        print("\nPositive-pixel percentage distribution:")

        print(
            f"  Minimum : {values.min():.6f}%"
        )

        print(
            f"  Maximum : {values.max():.6f}%"
        )

        print(
            f"  Mean    : {values.mean():.6f}%"
        )

        print(
            f"  Median  : {np.median(values):.6f}%"
        )

        print(
            f"  > 0%    : {np.sum(values > 0):,}"
        )

        print(
            f"  >= 0.1% : {np.sum(values >= 0.1):,}"
        )

        print(
            f"  >= 1%   : {np.sum(values >= 1.0):,}"
        )

        print(
            f"  >= 5%   : {np.sum(values >= 5.0):,}"
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("FORESTWATCH - PATCH DISTRIBUTION ANALYSIS")
print("=" * 80)

print(
    f"\nPatch size : {PATCH_SIZE} x {PATCH_SIZE}"
)

print(
    f"Stride     : {STRIDE}"
)

print(
    "\nNOTE: This is analysis only."
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
print("PATCH ANALYSIS COMPLETE")
print("=" * 80)