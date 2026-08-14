import os
import rasterio

RAW_FOLDER = "data/raw_corrected"

REGIONS = [
    "Chikkamagaluru",
    "Kodagu",
    "UttaraKannada"
]

YEARS = [2020, 2022, 2024]

DATASETS = [
    "Sentinel2",
    "NDVI",
    "Sentinel1",
    "RGB"
]


def compare_metadata(reference, current):
    """
    Compare the spatial metadata of two GeoTIFF files.
    """

    checks = {
        "CRS": reference.crs == current.crs,
        "Width": reference.width == current.width,
        "Height": reference.height == current.height,
        "Resolution": reference.res == current.res,
        "Transform": reference.transform == current.transform,
        "Bounds": reference.bounds == current.bounds,
    }

    return checks


print("=" * 80)
print("FORESTWATCH - SPATIAL ALIGNMENT CHECK")
print("=" * 80)

all_passed = True

for region in REGIONS:

    print("\n" + "=" * 80)
    print(f"REGION: {region}")
    print("=" * 80)

    for year in YEARS:

        print(f"\n--- {region} {year} ---")

        files = {}

        # --------------------------------------------------
        # Find all four files
        # --------------------------------------------------

        for dataset in DATASETS:

            filename = f"{region}_{dataset}_{year}.tif"
            filepath = os.path.join(RAW_FOLDER, filename)

            if not os.path.exists(filepath):

                print(f"❌ MISSING: {filename}")
                all_passed = False

            else:

                files[dataset] = filepath
                print(f"✓ Found: {filename}")

        # --------------------------------------------------
        # Cannot compare if a file is missing
        # --------------------------------------------------

        if len(files) != len(DATASETS):
            continue

        # --------------------------------------------------
        # Use Sentinel-2 as reference
        # --------------------------------------------------

        reference_file = files["Sentinel2"]

        with rasterio.open(reference_file) as reference:

            print("\nReference: Sentinel-2")

            print(f"  CRS        : {reference.crs}")
            print(f"  Size       : {reference.width} x {reference.height}")
            print(f"  Resolution : {reference.res}")
            print(f"  Bounds     : {reference.bounds}")

            # ------------------------------------------------
            # Compare every other dataset
            # ------------------------------------------------

            for dataset in DATASETS:

                if dataset == "Sentinel2":
                    continue

                filepath = files[dataset]

                with rasterio.open(filepath) as current:

                    checks = compare_metadata(
                        reference,
                        current
                    )

                    passed = all(checks.values())

                    print(f"\n{dataset}:")

                    for check, result in checks.items():

                        if result:
                            print(f"  ✓ {check}")
                        else:
                            print(f"  ❌ {check}")

                    if passed:

                        print(
                            f"  ✅ {dataset} is spatially aligned"
                        )

                    else:

                        print(
                            f"  ❌ {dataset} is NOT spatially aligned"
                        )

                        all_passed = False


print("\n" + "=" * 80)

if all_passed:

    print("✅ ALL SPATIAL ALIGNMENT CHECKS PASSED")

else:

    print("❌ SOME SPATIAL ALIGNMENT CHECKS FAILED")

print("=" * 80)