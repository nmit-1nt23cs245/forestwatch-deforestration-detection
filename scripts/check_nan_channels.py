import os
import numpy as np
import rasterio


FOLDER = "data/processed/normalized"

FILES = sorted([
    f for f in os.listdir(FOLDER)
    if f.endswith(".tif")
])


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
print("FORESTWATCH - NaN CHANNEL INSPECTION")
print("=" * 80)


for filename in FILES:

    filepath = os.path.join(
        FOLDER,
        filename
    )

    print("\n" + "-" * 80)
    print(filename)
    print("-" * 80)

    with rasterio.open(filepath) as src:

        data = src.read().astype(np.float32)

        total_nan = 0

        for i, channel_name in enumerate(CHANNEL_NAMES):

            channel = data[i]

            nan_count = np.isnan(channel).sum()

            total_nan += nan_count

            valid = channel[
                np.isfinite(channel)
            ]

            if len(valid) > 0:

                print(
                    f"{channel_name:5s} | "
                    f"NaN: {nan_count:,} | "
                    f"Min: {valid.min():.6f} | "
                    f"Max: {valid.max():.6f}"
                )

            else:

                print(
                    f"{channel_name:5s} | "
                    f"NaN: {nan_count:,} | "
                    f"NO VALID DATA"
                )

        print(
            f"\nTotal NaN values: {total_nan:,}"
        )


print("\n" + "=" * 80)
print("NaN INSPECTION COMPLETE")
print("=" * 80)