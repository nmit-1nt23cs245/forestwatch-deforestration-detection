import os
import numpy as np
import rasterio


INPUT_FOLDER = "data/processed/normalized"
OUTPUT_FOLDER = "data/processed/masks"

REGIONS = [
    "Chikkamagaluru",
    "Kodagu",
    "UttaraKannada"
]

YEARS = [
    2020,
    2022,
    2024
]


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("=" * 80)
print("FORESTWATCH - VALID DATA MASK GENERATION")
print("=" * 80)


for region in REGIONS:

    print("\n" + "-" * 80)
    print(f"REGION: {region}")
    print("-" * 80)

    year_masks = []
    reference_profile = None

    for year in YEARS:

        filename = f"{region}_{year}_7channel.tif"

        filepath = os.path.join(
            INPUT_FOLDER,
            filename
        )

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing file: {filepath}"
            )

        with rasterio.open(filepath) as src:

            data = src.read()

            if reference_profile is None:
                reference_profile = src.profile.copy()

            # ------------------------------------------------
            # Optical + NDVI channels:
            #
            # 0 = B2
            # 1 = B3
            # 2 = B4
            # 3 = B8
            # 4 = NDVI
            #
            # A pixel is valid only if ALL five are finite.
            # ------------------------------------------------

            optical = data[:5]

            valid_mask = np.all(
                np.isfinite(optical),
                axis=0
            )

            year_masks.append(valid_mask)

            print(
                f"{year}: "
                f"{valid_mask.sum():,} valid pixels"
            )

    # --------------------------------------------------------
    # Confirm masks are identical across all three years
    # --------------------------------------------------------

    same_2020_2022 = np.array_equal(
        year_masks[0],
        year_masks[1]
    )

    same_2022_2024 = np.array_equal(
        year_masks[1],
        year_masks[2]
    )

    if not same_2020_2022 or not same_2022_2024:

        raise ValueError(
            f"Validity masks differ between years "
            f"for {region}"
        )

    # Since all three masks are identical,
    # use the first one as the common mask.

    common_mask = year_masks[0]

    height, width = common_mask.shape

    total_pixels = height * width
    valid_pixels = int(common_mask.sum())
    invalid_pixels = total_pixels - valid_pixels

    valid_percentage = (
        valid_pixels / total_pixels
    ) * 100

    invalid_percentage = (
        invalid_pixels / total_pixels
    ) * 100

    print("\nCommon mask:")
    print(
        f"  Size             : "
        f"{width} x {height}"
    )

    print(
        f"  Valid pixels     : "
        f"{valid_pixels:,}"
    )

    print(
        f"  Invalid pixels   : "
        f"{invalid_pixels:,}"
    )

    print(
        f"  Valid percentage : "
        f"{valid_percentage:.4f}%"
    )

    print(
        f"  Invalid percentage : "
        f"{invalid_percentage:.4f}%"
    )

    # --------------------------------------------------------
    # Save mask
    #
    # 1 = valid
    # 0 = invalid
    # --------------------------------------------------------

    output_filename = (
        f"{region}_valid_mask.tif"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )

    output_profile = reference_profile.copy()

    output_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        compress="lzw",
        nodata=0
    )

    with rasterio.open(
        output_path,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            common_mask.astype(np.uint8),
            1
        )

    print(
        f"\nSaved: {output_path}"
    )


print("\n" + "=" * 80)
print("VALID MASK GENERATION COMPLETE")
print("=" * 80)