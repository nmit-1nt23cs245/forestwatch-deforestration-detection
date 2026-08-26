import csv
import os


METADATA_PATH = "data/processed/patches/patch_metadata.csv"


print("=" * 80)
print("FORESTWATCH - CLEAN PATCH METADATA")
print("=" * 80)


with open(
    METADATA_PATH,
    "r",
    newline=""
) as f:

    reader = csv.DictReader(f)
    rows = list(reader)

fieldnames = reader.fieldnames


unique_rows = []
seen = set()

for row in rows:

    key = (
        row["region"],
        row["period"],
        row["row"],
        row["column"]
    )

    if key in seen:
        continue

    seen.add(key)
    unique_rows.append(row)


print(
    f"\nOriginal rows : {len(rows)}"
)

print(
    f"Unique rows   : {len(unique_rows)}"
)

print(
    f"Removed       : "
    f"{len(rows) - len(unique_rows)}"
)


with open(
    METADATA_PATH,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        unique_rows
    )


print(
    "\n✅ Metadata cleaned successfully."
)

print("=" * 80)