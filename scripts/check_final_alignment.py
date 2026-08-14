import os
import rasterio

SATELLITE_FOLDER = "data/raw_corrected"
LABEL_FOLDER = "data/labels_aligned"

REGIONS = [
    "Chikkamagaluru",
    "Kodagu",
    "UttaraKannada"
]

PERIODS = [
    ("2020", "2020_2022"),
    ("2022", "2020_2022"),
    ("2022", "2022_2024"),
    ("2024", "2022_2024")
]


print("=" * 80)
print("FORESTWATCH - FINAL SATELLITE/LABEL ALIGNMENT CHECK")
print("=" * 80)

all_passed = True


for region in REGIONS:

    print("\n" + "=" * 80)
    print(f"REGION: {region}")
    print("=" * 80)

    for year, period in PERIODS:

        satellite_file = os.path.join(
            SATELLITE_FOLDER,
            f"{region}_Sentinel2_{year}.tif"
        )

        label_file = os.path.join(
            LABEL_FOLDER,
            f"{region}_Loss_{period}.tif"
        )

        print(f"\n--- {region} {year} / {period} ---")

        if not os.path.exists(satellite_file):
            print(f"❌ Missing satellite file: {satellite_file}")
            all_passed = False
            continue

        if not os.path.exists(label_file):
            print(f"❌ Missing label file: {label_file}")
            all_passed = False
            continue

        with rasterio.open(satellite_file) as sat, \
             rasterio.open(label_file) as label:

            checks = {
                "CRS": sat.crs == label.crs,
                "Width": sat.width == label.width,
                "Height": sat.height == label.height,
                "Resolution": sat.res == label.res,
                "Transform": sat.transform == label.transform,
                "Bounds": sat.bounds == label.bounds,
            }

            for name, result in checks.items():

                if result:
                    print(f"  ✓ {name}")
                else:
                    print(f"  ❌ {name}")
                    all_passed = False

            if all(checks.values()):
                print("  ✅ Satellite and label are perfectly aligned")
            else:
                print("  ❌ Alignment problem detected")


print("\n" + "=" * 80)

if all_passed:
    print("✅ FINAL ALIGNMENT CHECK PASSED")
else:
    print("❌ FINAL ALIGNMENT CHECK FAILED")

print("=" * 80)