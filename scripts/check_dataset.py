import os

SATELLITE_FOLDER = "data/raw_corrected"
LABEL_FOLDER = "data/labels_aligned"

REGIONS = [
    "Chikkamagaluru",
    "Kodagu",
    "UttaraKannada"
]

YEARS = [2020, 2022, 2024]

SATELLITE_PRODUCTS = [
    "Sentinel2",
    "NDVI",
    "Sentinel1",
    "RGB"
]

LABEL_PERIODS = [
    "2020_2022",
    "2022_2024"
]


print("=" * 80)
print("FORESTWATCH - COMPLETE DATASET INVENTORY CHECK")
print("=" * 80)


missing_files = []
found_files = []


# ============================================================
# 1. CHECK SATELLITE DATA
# ============================================================

print("\n" + "=" * 80)
print("SATELLITE DATA")
print("=" * 80)

for region in REGIONS:

    print(f"\n{region}")

    for year in YEARS:

        for product in SATELLITE_PRODUCTS:

            filename = f"{region}_{product}_{year}.tif"

            filepath = os.path.join(
                SATELLITE_FOLDER,
                filename
            )

            if os.path.exists(filepath):

                print(f"  ✓ {filename}")
                found_files.append(filepath)

            else:

                print(f"  ❌ MISSING: {filename}")
                missing_files.append(filepath)


# ============================================================
# 2. CHECK GROUND-TRUTH LABELS
# ============================================================

print("\n" + "=" * 80)
print("GROUND-TRUTH LABELS")
print("=" * 80)

for region in REGIONS:

    print(f"\n{region}")

    for period in LABEL_PERIODS:

        filename = f"{region}_Loss_{period}.tif"

        filepath = os.path.join(
            LABEL_FOLDER,
            filename
        )

        if os.path.exists(filepath):

            print(f"  ✓ {filename}")
            found_files.append(filepath)

        else:

            print(f"  ❌ MISSING: {filename}")
            missing_files.append(filepath)


# ============================================================
# 3. SUMMARY
# ============================================================

expected_satellite = (
    len(REGIONS)
    * len(YEARS)
    * len(SATELLITE_PRODUCTS)
)

expected_labels = (
    len(REGIONS)
    * len(LABEL_PERIODS)
)

expected_total = (
    expected_satellite
    + expected_labels
)


print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(f"Expected satellite files : {expected_satellite}")
print(f"Expected label files     : {expected_labels}")
print(f"Expected total files     : {expected_total}")

print(f"\nFiles found              : {len(found_files)}")
print(f"Files missing            : {len(missing_files)}")


# ============================================================
# 4. FINAL RESULT
# ============================================================

print("\n" + "=" * 80)

if len(missing_files) == 0:

    print("✅ COMPLETE DATASET FOUND")
    print("All 42 required files are present.")

else:

    print("❌ DATASET INCOMPLETE")

    print("\nMissing files:")

    for filepath in missing_files:
        print(f"  - {filepath}")

print("=" * 80)