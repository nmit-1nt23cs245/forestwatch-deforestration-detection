import os

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np


SATELLITE_FOLDER = "data/raw_corrected"
LABEL_FOLDER = "data/labels_raw"
OUTPUT_FOLDER = "data/labels_aligned"


REGIONS = [
    "Chikkamagaluru",
    "Kodagu",
    "UttaraKannada"
]

PERIODS = [
    "2020_2022",
    "2022_2024"
]


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("=" * 80)
print("FORESTWATCH - LABEL ALIGNMENT")
print("=" * 80)


for region in REGIONS:

    # ----------------------------------------------------------
    # Use Sentinel-2 2022 as the reference grid.
    # All Sentinel-2 years for the region have the same grid.
    # ----------------------------------------------------------

    reference_path = os.path.join(
        SATELLITE_FOLDER,
        f"{region}_Sentinel2_2022.tif"
    )

    if not os.path.exists(reference_path):

        print(f"\nERROR: Reference file missing:")
        print(reference_path)
        continue


    with rasterio.open(reference_path) as reference:

        print("\n" + "-" * 80)
        print(f"REGION: {region}")
        print("-" * 80)

        print(
            f"Reference grid: "
            f"{reference.width} x {reference.height}"
        )

        for period in PERIODS:

            label_filename = (
                f"{region}_Loss_{period}.tif"
            )

            input_path = os.path.join(
                LABEL_FOLDER,
                label_filename
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                label_filename
            )


            if not os.path.exists(input_path):

                print(
                    f"\nMISSING: {label_filename}"
                )

                continue


            # --------------------------------------------------
            # Read source label
            # --------------------------------------------------

            with rasterio.open(input_path) as source:

                source_data = source.read(1)


                # --------------------------------------------------
                # Create destination array
                # --------------------------------------------------

                destination = np.zeros(
                    (
                        reference.height,
                        reference.width
                    ),
                    dtype=np.uint8
                )


                # --------------------------------------------------
                # Reproject onto exact Sentinel-2 grid
                #
                # Nearest-neighbor is important because labels
                # are categorical: 0 or 1.
                # --------------------------------------------------

                reproject(
                    source=source_data,
                    destination=destination,

                    src_transform=source.transform,
                    src_crs=source.crs,

                    dst_transform=reference.transform,
                    dst_crs=reference.crs,

                    resampling=Resampling.nearest
                )


                # --------------------------------------------------
                # Output metadata
                # --------------------------------------------------

                profile = reference.profile.copy()

                profile.update(
                    driver="GTiff",
                    dtype=rasterio.uint8,
                    count=1,
                    compress="lzw",
                    nodata=0
                )


                # --------------------------------------------------
                # Write aligned label
                # --------------------------------------------------

                with rasterio.open(
                    output_path,
                    "w",
                    **profile
                ) as destination_file:

                    destination_file.write(
                        destination,
                        1
                    )


            positive_pixels = np.sum(
                destination == 1
            )

            total_pixels = destination.size

            percentage = (
                positive_pixels /
                total_pixels
            ) * 100


            print(
                f"\n{label_filename}"
            )

            print(
                f"  Output size     : "
                f"{reference.width} x "
                f"{reference.height}"
            )

            print(
                f"  Positive pixels : "
                f"{positive_pixels:,}"
            )

            print(
                f"  Positive area   : "
                f"{percentage:.4f}%"
            )

            print(
                f"  Saved to        : "
                f"{output_path}"
            )


print("\n" + "=" * 80)
print("LABEL ALIGNMENT COMPLETE")
print("=" * 80)