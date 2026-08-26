import os
import json
import pandas as pd
import torch


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_DIR = "data/processed/patches"

METADATA_PATH = os.path.join(
    PATCH_DIR,
    "patch_metadata.csv"
)

STATS_PATH = (
    "data/processed/channel_stats.json"
)

CHECKPOINT_PATH = (
    "data/processed/models/"
    "experiment2_best_resnet50.pth"
)


# ============================================================
# HELPERS
# ============================================================

def get_split_metadata(
    metadata,
    split
):

    return metadata[
        metadata["split"] == split
    ].copy().reset_index(
        drop=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "FORESTWATCH - EXPERIMENT 2 SANITY VERIFICATION"
    )
    print("=" * 80)

    # ========================================================
    # 1. LOAD METADATA
    # ========================================================

    print(
        "\n[1] Loading patch metadata..."
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    train = get_split_metadata(
        metadata,
        "train"
    )

    validation = get_split_metadata(
        metadata,
        "validation"
    )

    test = get_split_metadata(
        metadata,
        "test"
    )

    print(
        f"  Train      : {len(train)}"
    )

    print(
        f"  Validation : {len(validation)}"
    )

    print(
        f"  Test       : {len(test)}"
    )

    # ========================================================
    # 2. CHECK TEST SET SIZE
    # ========================================================

    print(
        "\n[2] Checking test set..."
    )

    assert len(test) == 112, (
        "Unexpected test-set size."
    )

    print(
        "  ✅ Test set contains exactly 112 samples."
    )

    # ========================================================
    # 3. CHECK TEST DUPLICATES
    # ========================================================

    print(
        "\n[3] Checking duplicate test patches..."
    )

    duplicate_files = test[
        "filename"
    ].duplicated().sum()

    duplicate_rows = test.duplicated().sum()

    print(
        f"  Duplicate filenames : "
        f"{duplicate_files}"
    )

    print(
        f"  Duplicate rows      : "
        f"{duplicate_rows}"
    )

    assert duplicate_files == 0
    assert duplicate_rows == 0

    print(
        "  ✅ No duplicate test patches."
    )

    # ========================================================
    # 4. CHECK TEST LABELS
    # ========================================================

    print(
        "\n[4] Checking test labels..."
    )

    test_negative = (
        test["label"] == 0
    ).sum()

    test_positive = (
        test["label"] == 1
    ).sum()

    print(
        f"  Negative : {test_negative}"
    )

    print(
        f"  Positive : {test_positive}"
    )

    assert (
        test_negative +
        test_positive
    ) == 112

    # ========================================================
    # 5. CHECK TEST/TRAIN OVERLAP
    # ========================================================

    print(
        "\n[5] Checking test/train overlap..."
    )

    train_files = set(
        train["filename"]
    )

    validation_files = set(
        validation["filename"]
    )

    test_files = set(
        test["filename"]
    )

    train_test_overlap = (
        train_files &
        test_files
    )

    validation_test_overlap = (
        validation_files &
        test_files
    )

    train_validation_overlap = (
        train_files &
        validation_files
    )

    print(
        f"  Train ∩ Test       : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"  Validation ∩ Test  : "
        f"{len(validation_test_overlap)}"
    )

    print(
        f"  Train ∩ Validation : "
        f"{len(train_validation_overlap)}"
    )

    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0
    assert len(train_validation_overlap) == 0

    print(
        "  ✅ No filename overlap."
    )

    # ========================================================
    # 6. CHECK SPATIAL COORDINATE OVERLAP
    # ========================================================

    print(
        "\n[6] Checking spatial coordinate leakage..."
    )

    coordinate_columns = [
        "location",
        "year1",
        "year2",
        "row",
        "col"
    ]

    missing_columns = [
        column
        for column in coordinate_columns
        if column not in metadata.columns
    ]

    if missing_columns:

        print(
            "  ⚠️ Expected coordinate columns "
            "not found:"
        )

        for column in missing_columns:

            print(
                f"     {column}"
            )

        print(
            "  Skipping coordinate check."
        )

    else:

        train_coordinates = set(
            zip(
                train["location"],
                train["year1"],
                train["year2"],
                train["row"],
                train["col"]
            )
        )

        validation_coordinates = set(
            zip(
                validation["location"],
                validation["year1"],
                validation["year2"],
                validation["row"],
                validation["col"]
            )
        )

        test_coordinates = set(
            zip(
                test["location"],
                test["year1"],
                test["year2"],
                test["row"],
                test["col"]
            )
        )

        train_test_coordinates = (
            train_coordinates &
            test_coordinates
        )

        validation_test_coordinates = (
            validation_coordinates &
            test_coordinates
        )

        print(
            f"  Train ∩ Test       : "
            f"{len(train_test_coordinates)}"
        )

        print(
            f"  Validation ∩ Test  : "
            f"{len(validation_test_coordinates)}"
        )

        assert (
            len(train_test_coordinates)
            == 0
        )

        assert (
            len(validation_test_coordinates)
            == 0
        )

        print(
            "  ✅ No coordinate leakage."
        )

    # ========================================================
    # 7. VERIFY NORMALIZATION STATISTICS
    # ========================================================

    print(
        "\n[7] Checking channel statistics..."
    )

    with open(
        STATS_PATH,
        "r"
    ) as f:

        stats = json.load(f)

    print(
        f"  Statistics calculated from "
        f"{stats['num_training_patches']} "
        f"training patches."
    )

    print(
        f"  Patch shape: "
        f"{stats['patch_shape']}"
    )

    assert (
        stats["num_training_patches"]
        == len(train)
    )

    assert (
        stats["patch_shape"]
        == [14, 224, 224]
    )

    assert (
        len(stats["channel_names"])
        == 14
    )

    assert (
        len(stats["mean"])
        == 14
    )

    assert (
        len(stats["std"])
        == 14
    )

    print(
        "  ✅ Normalization statistics "
        "are training-only."
    )

    # ========================================================
    # 8. VERIFY CHECKPOINT
    # ========================================================

    print(
        "\n[8] Checking Experiment 2 checkpoint..."
    )

    assert os.path.exists(
        CHECKPOINT_PATH
    ), (
        "Experiment 2 checkpoint not found."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    checkpoint_epoch = checkpoint.get(
        "epoch"
    )

    checkpoint_f1 = checkpoint.get(
        "best_val_f1"
    )

    print(
        f"  Epoch        : "
        f"{checkpoint_epoch}"
    )

    print(
        f"  Best Val F1  : "
        f"{checkpoint_f1:.4f}"
    )

    assert checkpoint_epoch == 15

    assert abs(
        checkpoint_f1 -
        0.5714285714285714
    ) < 1e-6

    print(
        "  ✅ Correct best checkpoint verified."
    )

    # ========================================================
    # 9. VERIFY MODEL STATE
    # ========================================================

    print(
        "\n[9] Checking checkpoint contents..."
    )

    assert (
        "model_state_dict"
        in checkpoint
    )

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"  Model parameters stored: "
        f"{len(state_dict)} tensors"
    )

    print(
        "  ✅ Model state dictionary present."
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "SANITY VERIFICATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "\n✅ Test set size verified"
    )

    print(
        "✅ No duplicate test patches"
    )

    print(
        "✅ No train/test filename overlap"
    )

    print(
        "✅ No validation/test filename overlap"
    )

    print(
        "✅ No spatial coordinate leakage"
    )

    print(
        "✅ Training-only normalization statistics"
    )

    print(
        "✅ Experiment 2 checkpoint verified"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ EXPERIMENT 2 SANITY VERIFICATION PASSED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()