import os
import numpy as np
import rasterio


DATA_FOLDER = "data/processed/normalized"

FILES = [
    "Kodagu_2020_7channel.tif",
    "Kodagu_2022_7channel.tif",
    "Kodagu_2024_7channel.tif"
]

CHANNEL_NAMES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "NDVI",
    "VV",
    "VH"
]


print("=" * 80)
print("FORESTWATCH - SOURCE NODATA DIAGNOSTIC")
print("=" * 80)


for filename in FILES:

    path = os.path.join(
        DATA_FOLDER,
        filename
    )

    print("\n" + "=" * 80)
    print(filename)
    print("=" * 80)

    if not os.path.exists(path):

        print(
            f"❌ File not found: {path}"
        )

        continue

    with rasterio.open(path) as src:

        print(
            f"Size       : {src.width} x {src.height}"
        )

        print(
            f"Bands      : {src.count}"
        )

        print(
            f"Nodata     : {src.nodata}"
        )

        print(
            f"Data type  : {src.dtypes}"
        )

        for band_index in range(1, src.count + 1):

            data = src.read(
                band_index
            )

            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()

            finite = data[
                np.isfinite(data)
            ]

            if len(finite) > 0:

                min_value = finite.min()
                max_value = finite.max()

            else:

                min_value = np.nan
                max_value = np.nan

            print(
                f"\n  {CHANNEL_NAMES[band_index - 1]}"
            )

            print(
                f"    NaN count : {nan_count:,}"
            )

            print(
                f"    Inf count : {inf_count:,}"
            )

            print(
                f"    Min       : {min_value}"
            )

            print(
                f"    Max       : {max_value}"
            )


print("\n" + "=" * 80)
print("SOURCE NODATA DIAGNOSTIC COMPLETE")
print("=" * 80)