import os
import rasterio
import numpy as np

LABEL_FOLDER = "data/labels_raw"

print("=" * 80)
print("FORESTWATCH - GROUND TRUTH LABEL INSPECTION")
print("=" * 80)

tif_files = sorted([
    f for f in os.listdir(LABEL_FOLDER)
    if f.lower().endswith((".tif", ".tiff"))
])

print(f"\nTotal label files found: {len(tif_files)}\n")

for filename in tif_files:

    filepath = os.path.join(LABEL_FOLDER, filename)

    print("-" * 80)
    print(f"FILE: {filename}")
    print("-" * 80)

    try:

        with rasterio.open(filepath) as src:

            print(f"Width          : {src.width}")
            print(f"Height         : {src.height}")
            print(f"Bands          : {src.count}")
            print(f"CRS            : {src.crs}")
            print(f"Resolution     : {src.res}")
            print(f"Data type      : {src.dtypes}")
            print(f"NoData         : {src.nodata}")
            print(f"Bounds         : {src.bounds}")

            data = src.read(1, masked=True)

            valid_data = data.compressed()
            valid_data = valid_data[np.isfinite(valid_data)]

            if len(valid_data) > 0:

                unique_values, counts = np.unique(
                    valid_data,
                    return_counts=True
                )

                print("\nUnique values:")

                for value, count in zip(
                    unique_values,
                    counts
                ):

                    percentage = (
                        count / len(valid_data)
                    ) * 100

                    print(
                        f"  {value} : "
                        f"{count:,} pixels "
                        f"({percentage:.4f}%)"
                    )

                positive_pixels = np.sum(
                    valid_data == 1
                )

                positive_percentage = (
                    positive_pixels /
                    len(valid_data)
                ) * 100

                print(
                    f"\nForest-loss pixels : "
                    f"{positive_pixels:,}"
                )

                print(
                    f"Forest-loss area   : "
                    f"{positive_percentage:.4f}%"
                )

            else:

                print("No valid pixels found.")

    except Exception as e:

        print(f"ERROR: {e}")


print("\n" + "=" * 80)
print("LABEL INSPECTION COMPLETE")
print("=" * 80)