import os
import csv
import numpy as np


# ============================================================
# FORESTWATCH - PATCH DATASET VERIFICATION
#
# Verifies:
#   1. Patch shape = (14, 224, 224)
#   2. No NaN values
#   3. No infinite values
#   4. Labels are only 0 / 1
#   5. Metadata matches saved patches
#   6. No duplicate metadata entries
#   7. Split counts
#   8. Spatial coordinates do not cross splits
# ============================================================


PATCH_FOLDER = "data/processed/patches"

SPLITS = [
    "train",
    "validation",
    "test"
]

EXPECTED_SHAPE = (
    14,
    224,
    224
)

METADATA_PATH = os.path.join(
    PATCH_FOLDER,
    "patch_metadata.csv"
)


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("FORESTWATCH - PATCH DATASET VERIFICATION")
print("=" * 80)


# ============================================================
# 1. Check folders
# ============================================================

print("\n[1] Checking dataset folders...")

for split in SPLITS:

    folder = os.path.join(
        PATCH_FOLDER,
        split
    )

    if not os.path.isdir(folder):

        raise FileNotFoundError(
            f"Missing split folder: {folder}"
        )

    print(
        f"  {split:12s}: OK"
    )


# ============================================================
# 2. Load metadata
# ============================================================

print("\n[2] Loading metadata...")

if not os.path.exists(METADATA_PATH):

    raise FileNotFoundError(
        f"Missing metadata: {METADATA_PATH}"
    )

with open(
    METADATA_PATH,
    "r",
    newline=""
) as f:

    reader = csv.DictReader(f)

    metadata = list(reader)


print(
    f"  Metadata rows: {len(metadata)}"
)


# ============================================================
# 3. Check duplicate metadata
# ============================================================

print("\n[3] Checking duplicate metadata...")

metadata_keys = []

for row in metadata:

    key = (
        row["region"],
        row["period"],
        row["row"],
        row["column"]
    )

    metadata_keys.append(key)


duplicate_keys = (
    len(metadata_keys)
    -
    len(set(metadata_keys))
)


print(
    f"  Duplicate entries: {duplicate_keys}"
)

if duplicate_keys > 0:

    print(
        "  ❌ Duplicate metadata entries detected."
    )

else:

    print(
        "  ✅ No duplicate metadata entries."
    )


# ============================================================
# 4. Verify patches
# ============================================================

print("\n[4] Verifying .npy patches...")

total_files = 0
bad_shape = 0
nan_files = 0
inf_files = 0
read_errors = 0

split_counts = {
    "train": 0,
    "validation": 0,
    "test": 0
}

for split in SPLITS:

    folder = os.path.join(
        PATCH_FOLDER,
        split
    )

    files = [
        f
        for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    print(
        f"\n  {split.upper()}"
    )

    print(
        f"    Files: {len(files)}"
    )

    split_counts[split] = len(files)

    for filename in files:

        filepath = os.path.join(
            folder,
            filename
        )

        total_files += 1

        try:

            patch = np.load(
                filepath
            )

        except Exception as e:

            print(
                f"    ❌ Read error: "
                f"{filename}"
            )

            read_errors += 1

            continue

        # Shape
        if patch.shape != EXPECTED_SHAPE:

            bad_shape += 1

            if bad_shape <= 10:

                print(
                    f"    ❌ Bad shape: "
                    f"{filename} "
                    f"{patch.shape}"
                )

        # NaN
        if np.isnan(patch).any():

            nan_files += 1

            if nan_files <= 10:

                print(
                    f"    ❌ NaN detected: "
                    f"{filename}"
                )

        # Inf
        if np.isinf(patch).any():

            inf_files += 1

            if inf_files <= 10:

                print(
                    f"    ❌ Inf detected: "
                    f"{filename}"
                )


# ============================================================
# 5. Verify metadata filenames
# ============================================================

print("\n[5] Checking metadata ↔ patch files...")

actual_files = set()

for split in SPLITS:

    folder = os.path.join(
        PATCH_FOLDER,
        split
    )

    for filename in os.listdir(folder):

        if filename.endswith(".npy"):

            actual_files.add(
                filename
            )


metadata_files = set(
    row["filename"]
    for row in metadata
)


missing_from_metadata = (
    actual_files -
    metadata_files
)

missing_from_disk = (
    metadata_files -
    actual_files
)


print(
    f"  Files on disk     : "
    f"{len(actual_files)}"
)

print(
    f"  Files in metadata : "
    f"{len(metadata_files)}"
)

print(
    f"  Missing metadata  : "
    f"{len(missing_from_metadata)}"
)

print(
    f"  Missing on disk   : "
    f"{len(missing_from_disk)}"
)


# ============================================================
# 6. Verify labels
# ============================================================

print("\n[6] Checking labels...")

invalid_labels = 0

label_counts = {
    "train": {
        0: 0,
        1: 0
    },
    "validation": {
        0: 0,
        1: 0
    },
    "test": {
        0: 0,
        1: 0
    }
}


for row in metadata:

    label = int(
        row["label"]
    )

    split = row["split"]

    if label not in [0, 1]:

        invalid_labels += 1

    elif split in label_counts:

        label_counts[split][label] += 1


for split in SPLITS:

    print(
        f"  {split:12s}: "
        f"negative={label_counts[split][0]}, "
        f"positive={label_counts[split][1]}"
    )


# ============================================================
# 7. Check spatial leakage
# ============================================================

print("\n[7] Checking spatial split leakage...")

spatial_assignments = {}

leakage_count = 0

for row in metadata:

    key = (
        row["region"],
        row["period"],
        row["row"],
        row["column"]
    )

    split = row["split"]

    if key in spatial_assignments:

        previous_split = spatial_assignments[key]

        if previous_split != split:

            leakage_count += 1

            print(
                "  ❌ Spatial leakage:"
            )

            print(
                f"     {key}"
            )

            print(
                f"     {previous_split} "
                f"vs {split}"
            )

    else:

        spatial_assignments[key] = split


print(
    f"  Conflicting coordinates: "
    f"{leakage_count}"
)


# ============================================================
# 8. Verify 14-channel structure
# ============================================================

print("\n[8] Checking 14-channel structure...")

channel_errors = 0

sample_checked = 0

for split in SPLITS:

    folder = os.path.join(
        PATCH_FOLDER,
        split
    )

    files = [
        f
        for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    for filename in files[:5]:

        patch = np.load(
            os.path.join(
                folder,
                filename
            )
        )

        if patch.shape[0] != 14:

            channel_errors += 1

        sample_checked += 1


print(
    f"  Samples checked: "
    f"{sample_checked}"
)

print(
    f"  Channel errors : "
    f"{channel_errors}"
)


# ============================================================
# 9. Final verification
# ============================================================

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

print(
    f"\nTotal patch files : {total_files}"
)

print(
    f"Metadata rows     : {len(metadata)}"
)

print(
    f"Bad shapes        : {bad_shape}"
)

print(
    f"NaN files         : {nan_files}"
)

print(
    f"Inf files         : {inf_files}"
)

print(
    f"Read errors       : {read_errors}"
)

print(
    f"Invalid labels    : {invalid_labels}"
)

print(
    f"Spatial leakage   : {leakage_count}"
)

print(
    f"Channel errors    : {channel_errors}"
)

print(
    f"Metadata mismatch : "
    f"{len(missing_from_metadata) + len(missing_from_disk)}"
)


# ============================================================
# FINAL RESULT
# ============================================================

if (
    bad_shape == 0
    and
    nan_files == 0
    and
    inf_files == 0
    and
    read_errors == 0
    and
    invalid_labels == 0
    and
    leakage_count == 0
    and
    channel_errors == 0
    and
    len(missing_from_metadata) == 0
    and
    len(missing_from_disk) == 0
    and
    duplicate_keys == 0
):

    print(
        "\n✅ DATASET VERIFICATION PASSED"
    )

    print(
        "\nThe dataset is ready for model development."
    )

else:

    print(
        "\n❌ DATASET VERIFICATION FAILED"
    )

    print(
        "\nFix the reported issue(s) before training."
    )


print("\n" + "=" * 80)