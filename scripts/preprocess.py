import os

import numpy as np
import rasterio


# ============================================================
# FORESTWATCH - STAGE 1 PREPROCESSING
#
# Creates 7-channel yearly data:
#
# B2, B3, B4, B8, NDVI, VV, VH
#
# This script:
#   1. Reads scientific GeoTIFFs
#   2. Scales Sentinel-2 reflectance
#   3. Cleans invalid values
#   4. Keeps NDVI numerical
#   5. Keeps SAR values in dB
#   6. Saves a 7-channel GeoTIFF
#
# IMPORTANT:
# SAR normalization is NOT performed here.
# Its statistics will be calculated later using training data only.
# ============================================================


INPUT_FOLDER = "data/raw_corrected"

OUTPUT_FOLDER = "data/processed/normalized"

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


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


print("=" * 80)
print("FORESTWATCH - STAGE 1 PREPROCESSING")
print("=" * 80)


def read_single_band(filepath, band_number=1):

    with rasterio.open(filepath) as src:

        data = src.read(
            band_number
        ).astype(np.float32)

        profile = src.profile.copy()

        return data, profile


def read_sentinel2(filepath):

    """
    Sentinel-2 bands:

    Band 1 -> B2
    Band 2 -> B3
    Band 3 -> B4
    Band 4 -> B8
    """

    with rasterio.open(filepath) as src:

        data = src.read().astype(
            np.float32
        )

        profile = src.profile.copy()

    return data, profile


def preprocess_sentinel2(data):

    """
    Sentinel-2 SR values are stored using
    a scale factor of approximately 10000.

    Convert to reflectance.
    """

    data = data / 10000.0

    # Remove physically unreasonable values
    data = np.clip(
        data,
        0.0,
        1.0
    )

    return data


def preprocess_ndvi(data):

    """
    NDVI should remain a numerical vegetation index.
    """

    data = np.clip(
        data,
        -1.0,
        1.0
    )

    return data


def clean_sar(data):

    """
    Keep Sentinel-1 VV/VH in dB for now.

    Normalization will be performed later using
    training-set statistics only.
    """

    # Replace NaN / infinite values
    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return data


# ============================================================
# PROCESS EACH REGION AND YEAR
# ============================================================

for region in REGIONS:

    for year in YEARS:

        print("\n" + "-" * 80)
        print(f"Processing: {region} - {year}")
        print("-" * 80)


        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        s2_path = os.path.join(
            INPUT_FOLDER,
            f"{region}_Sentinel2_{year}.tif"
        )

        ndvi_path = os.path.join(
            INPUT_FOLDER,
            f"{region}_NDVI_{year}.tif"
        )

        s1_path = os.path.join(
            INPUT_FOLDER,
            f"{region}_Sentinel1_{year}.tif"
        )


        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        required_files = [
            s2_path,
            ndvi_path,
            s1_path
        ]

        missing = [
            path
            for path in required_files
            if not os.path.exists(path)
        ]

        if missing:

            print("ERROR: Missing files:")

            for path in missing:
                print(f"  {path}")

            continue


        # ----------------------------------------------------
        # Sentinel-2
        # ----------------------------------------------------

        print("Reading Sentinel-2...")

        s2, profile = read_sentinel2(
            s2_path
        )

        print(
            f"  Shape before preprocessing: "
            f"{s2.shape}"
        )

        s2 = preprocess_sentinel2(
            s2
        )

        print(
            f"  Shape after preprocessing: "
            f"{s2.shape}"
        )


        # ----------------------------------------------------
        # NDVI
        # ----------------------------------------------------

        print("Reading NDVI...")

        ndvi, _ = read_single_band(
            ndvi_path
        )

        ndvi = preprocess_ndvi(
            ndvi
        )

        print(
            f"  NDVI shape: {ndvi.shape}"
        )


        # ----------------------------------------------------
        # Sentinel-1
        # ----------------------------------------------------

        print("Reading Sentinel-1...")

        s1, _ = read_sentinel2(
            s1_path
        )

        s1 = clean_sar(
            s1
        )

        print(
            f"  SAR shape: {s1.shape}"
        )


        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        expected_height = s2.shape[1]
        expected_width = s2.shape[2]


        if ndvi.shape != (
            expected_height,
            expected_width
        ):

            raise ValueError(
                f"NDVI dimension mismatch for "
                f"{region} {year}: "
                f"{ndvi.shape} vs "
                f"{(expected_height, expected_width)}"
            )


        if s1.shape[1:] != (
            expected_height,
            expected_width
        ):

            raise ValueError(
                f"SAR dimension mismatch for "
                f"{region} {year}: "
                f"{s1.shape[1:]} vs "
                f"{(expected_height, expected_width)}"
            )


        # ----------------------------------------------------
        # Combine into 7 channels
        #
        # Channel order:
        #
        # 0 = B2
        # 1 = B3
        # 2 = B4
        # 3 = B8
        # 4 = NDVI
        # 5 = VV
        # 6 = VH
        # ----------------------------------------------------

        combined = np.concatenate(
            [
                s2,
                ndvi[np.newaxis, :, :],
                s1
            ],
            axis=0
        )


        print(
            f"Combined shape: "
            f"{combined.shape}"
        )


        # ----------------------------------------------------
        # Check for invalid values
        # ----------------------------------------------------

        nan_count = np.isnan(
            combined
        ).sum()

        inf_count = np.isinf(
            combined
        ).sum()

        print(
            f"NaN values : {nan_count}"
        )

        print(
            f"Inf values : {inf_count}"
        )


        # ----------------------------------------------------
        # Update GeoTIFF metadata
        # ----------------------------------------------------

        output_profile = profile.copy()

        output_profile.update(
            driver="GTiff",
            dtype="float32",
            count=7,
            compress="lzw"
        )


        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        output_filename = (
            f"{region}_{year}_7channel.tif"
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_filename
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        with rasterio.open(
            output_path,
            "w",
            **output_profile
        ) as dst:

            dst.write(
                combined
            )


        print(
            f"Saved: {output_path}"
        )


print("\n" + "=" * 80)
print("STAGE 1 PREPROCESSING COMPLETE")
print("=" * 80)
