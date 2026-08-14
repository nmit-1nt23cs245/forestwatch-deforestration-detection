import os
import rasterio
import numpy as np

RAW_FOLDER = "data/raw_corrected"

print("=" * 80)
print("FORESTWATCH - GEOTIFF METADATA INSPECTION")
print("=" * 80)

tif_files = sorted([
    f for f in os.listdir(RAW_FOLDER)
    if f.lower().endswith((".tif", ".tiff"))
])

print(f"\nTotal TIFF files found: {len(tif_files)}\n")

for filename in tif_files:

    filepath = os.path.join(RAW_FOLDER, filename)

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

            # Calculate value statistics for each band
            for band in range(1, src.count + 1):

                data = src.read(band, masked=True)

                # Convert masked array to normal array containing
                # only unmasked/valid values
                valid_data = data.compressed()

                # Remove NaN and infinite values
                valid_data = valid_data[np.isfinite(valid_data)]

                if len(valid_data) > 0:

                    print(
                        f"Band {band} range  : "
                        f"{valid_data.min():.6f} to "
                        f"{valid_data.max():.6f}"
                    )

                    print(
                        f"Band {band} mean   : "
                        f"{valid_data.mean():.6f}"
                    )

                    print(
                        f"Band {band} valid pixels : "
                        f"{len(valid_data)}"
                    )

                else:

                    print(
                        f"Band {band} : No finite valid values"
                    )

    except Exception as e:

        print(f"ERROR: {e}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)