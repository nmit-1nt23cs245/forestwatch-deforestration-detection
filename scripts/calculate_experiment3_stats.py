import os
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_DIR = "data/processed/patches"

METADATA_PATH = os.path.join(
    PATCH_DIR,
    "patch_metadata.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "experiment3_channel_stats.json"
)

CHANNEL_NAMES_14 = [
    "Year1_B2",
    "Year1_B3",
    "Year1_B4",
    "Year1_B8",
    "Year1_NDVI",
    "Year1_VV",
    "Year1_VH",

    "Year2_B2",
    "Year2_B3",
    "Year2_B4",
    "Year2_B8",
    "Year2_NDVI",
    "Year2_VV",
    "Year2_VH"
]

DELTA_NAMES = [
    "Delta_B2",
    "Delta_B3",
    "Delta_B4",
    "Delta_B8",
    "Delta_NDVI",
    "Delta_VV",
    "Delta_VH"
]

CHANNEL_NAMES_21 = (
    CHANNEL_NAMES_14 +
    DELTA_NAMES
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "FORESTWATCH - EXPERIMENT 3 CHANNEL STATISTICS"
    )
    print("=" * 80)

    print(
        "\nInput:"
    )

    print(
        "  Existing patches : 14 channels"
    )

    print(
        "  Temporal delta   : 7 channels"
    )

    print(
        "  Final input      : 21 channels"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_PATH
    )

    train_metadata = metadata[
        metadata["split"] == "train"
    ].copy()

    print(
        f"\nTraining patches: "
        f"{len(train_metadata)}"
    )

    # --------------------------------------------------------
    # Running statistics
    #
    # We use sum and sum of squares instead of storing
    # every pixel from every patch in memory.
    # --------------------------------------------------------

    num_channels = 21

    sums = np.zeros(
        num_channels,
        dtype=np.float64
    )

    squared_sums = np.zeros(
        num_channels,
        dtype=np.float64
    )

    counts = np.zeros(
        num_channels,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Process training patches only
    # --------------------------------------------------------

    for index, row in enumerate(
        train_metadata.itertuples(index=False),
        start=1
    ):

        filepath = os.path.join(
            PATCH_DIR,
            row.split,
            row.filename
        )

        patch = np.load(
            filepath
        ).astype(
            np.float64
        )

        # ----------------------------------------------------
        # Verify existing 14-channel structure
        # ----------------------------------------------------

        if patch.shape != (
            14,
            224,
            224
        ):

            raise ValueError(
                f"Unexpected patch shape "
                f"{patch.shape} in "
                f"{row.filename}"
            )

        # ----------------------------------------------------
        # Split temporal observations
        # ----------------------------------------------------

        year1 = patch[:7]

        year2 = patch[7:14]

        # ----------------------------------------------------
        # Calculate temporal difference
        #
        # Delta = Year2 - Year1
        # ----------------------------------------------------

        delta = (
            year2 -
            year1
        )

        # ----------------------------------------------------
        # Combine into 21-channel representation
        # ----------------------------------------------------

        patch_21 = np.concatenate(
            [
                year1,
                year2,
                delta
            ],
            axis=0
        )

        if patch_21.shape != (
            21,
            224,
            224
        ):

            raise ValueError(
                f"Unexpected 21-channel "
                f"shape: {patch_21.shape}"
            )

        # ----------------------------------------------------
        # Calculate statistics
        # ----------------------------------------------------

        for channel in range(
            num_channels
        ):

            values = patch_21[
                channel
            ].reshape(-1)

            valid = np.isfinite(
                values
            )

            values = values[
                valid
            ]

            if values.size == 0:

                continue

            sums[channel] += (
                values.sum()
            )

            squared_sums[channel] += (
                np.square(values).sum()
            )

            counts[channel] += (
                values.size
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 25 == 0
            or index == len(train_metadata)
        ):

            print(
                f"  Processed "
                f"{index}/"
                f"{len(train_metadata)} patches"
            )

    # --------------------------------------------------------
    # Calculate mean and standard deviation
    # --------------------------------------------------------

    means = (
        sums /
        counts
    )

    variances = (
        squared_sums /
        counts
    ) - np.square(
        means
    )

    # Numerical safety
    variances = np.maximum(
        variances,
        0.0
    )

    stds = np.sqrt(
        variances
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "21-CHANNEL STATISTICS"
    )

    print(
        "=" * 80
    )

    for index, name in enumerate(
        CHANNEL_NAMES_21
    ):

        print(
            f"{name:<15} "
            f"Mean = {means[index]:12.6f}   "
            f"Std = {stds[index]:12.6f}"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {

        "channel_names": (
            CHANNEL_NAMES_21
        ),

        "mean": (
            means.tolist()
        ),

        "std": (
            stds.tolist()
        ),

        "num_training_patches": int(
            len(train_metadata)
        ),

        "patch_shape": [
            21,
            224,
            224
        ],

        "construction": (
            "Year1(7) + Year2(7) + "
            "(Year2-Year1)(7)"
        ),

        "delta_definition": (
            "Delta = Year2 - Year1"
        )
    }

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    print(
        "\nSaved statistics to:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print(
        "\n✅ Experiment 3 channel "
        "statistics calculated successfully."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()