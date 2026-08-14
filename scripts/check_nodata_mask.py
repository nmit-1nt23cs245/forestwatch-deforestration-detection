import os
import numpy as np
import rasterio


FOLDER = "data/processed/normalized"

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


print("=" * 80)
print("FORESTWATCH - OPTICAL NODATA MASK CHECK")
print("=" * 80)


for region in REGIONS:

    print("\n" + "=" * 80)
    print(f"REGION: {region}")
    print("=" * 80)

    masks = []

    for year in YEARS:

        filename = (
            f"{region}_{year}_7channel.tif"
        )

        filepath = os.path.join(
            FOLDER,
            filename
        )

        with rasterio.open(filepath) as src:

            # B2 is channel 1.
            # Since the NaN pattern is shared by
            # all optical channels, use B2 as mask.
            b2 = src.read(1)

            invalid_mask = np.isnan(b2)

            masks.append(invalid_mask)

            print(
                f"{year}: "
                f"{invalid_mask.sum():,} "
                f"invalid pixels"
            )

    mask_2020 = masks[0]
    mask_2022 = masks[1]
    mask_2024 = masks[2]

    same_2020_2022 = np.array_equal(
        mask_2020,
        mask_2022
    )

    same_2022_2024 = np.array_equal(
        mask_2022,
        mask_2024
    )

    same_all = np.array_equal(
        mask_2020,
        mask_2022
    ) and np.array_equal(
        mask_2022,
        mask_2024
    )

    print("\nMask comparison:")

    print(
        f"2020 == 2022 : "
        f"{same_2020_2022}"
    )

    print(
        f"2022 == 2024 : "
        f"{same_2022_2024}"
    )

    print(
        f"All three identical : "
        f"{same_all}"
    )

    # --------------------------------------------------------
    # Count pixels valid in every year
    # --------------------------------------------------------

    valid_all_years = (
        ~mask_2020
        & ~mask_2022
        & ~mask_2024
    )

    print(
        f"\nPixels valid in all 3 years: "
        f"{valid_all_years.sum():,}"
    )

    print(
        f"Pixels invalid in at least one year: "
        f"{(~valid_all_years).sum():,}"
    )


print("\n" + "=" * 80)
print("NODATA MASK CHECK COMPLETE")
print("=" * 80)