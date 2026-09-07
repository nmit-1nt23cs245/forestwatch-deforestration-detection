# ForestWatch – Multi-Temporal Satellite Image Analysis for Deforestation Detection

> **An AI-based multi-sensor and multi-temporal satellite image classification system for detecting potential forest-cover loss using Sentinel-1 SAR, Sentinel-2 optical data, NDVI, temporal-difference features, and a modified ResNet50.**

---

## 📌 Project Overview

**ForestWatch** is a deep-learning-based satellite image analysis project designed to identify potential forest-cover loss from multi-temporal satellite observations.

The project combines complementary information from:

- **Sentinel-2 optical imagery**
- **NDVI (Normalized Difference Vegetation Index)**
- **Sentinel-1 SAR (VV and VH)**

A major challenge in satellite-based deforestation detection is that **seasonal vegetation changes and other normal land-cover variations can resemble actual forest loss**.

To address this challenge, ForestWatch progressively evaluates different input representations using the same **ResNet50 backbone**:

1. **Experiment 1 – 14-channel baseline**
2. **Experiment 2 – 14-channel model with regularization and augmentation**
3. **Experiment 3 – 21-channel temporal-difference representation**

The final experiment explicitly provides the model with:

- the earlier observation,
- the later observation, and
- the difference between them.

This gives the network direct information about how the satellite-derived features changed over time.

---

# 🎯 Problem Statement

Satellite imagery provides an effective way to monitor large forest regions, but automated deforestation detection is challenging because:

- Seasonal vegetation changes can resemble forest loss.
- Agricultural cycles can create strong temporal changes.
- Optical imagery can be affected by cloud cover.
- Different land-cover types can have similar spectral characteristics.
- Forest structure and vegetation condition are not completely represented by a single sensor.
- Small or fragmented changes can be difficult to distinguish from normal variation.

### Research Problem

> **Existing image-based approaches can become confused between normal/seasonal vegetation variation and actual forest-cover loss.**

ForestWatch investigates whether combining **multi-sensor information** with an explicit **multi-temporal change representation** can improve forest/non-forest classification on the project's held-out test data.

---

# 💡 Proposed Approach

ForestWatch uses a multi-sensor, multi-temporal representation.

## 7 Features per Observation

Each temporal observation contains **7 channels**:

| Channel | Source | Description |
|---|---|---|
| B2 | Sentinel-2 | Blue |
| B3 | Sentinel-2 | Green |
| B4 | Sentinel-2 | Red |
| B8 | Sentinel-2 | Near Infrared |
| NDVI | Derived from Sentinel-2 | Vegetation index |
| VV | Sentinel-1 | VV polarization |
| VH | Sentinel-1 | VH polarization |

### NDVI

NDVI is calculated as:

```text
NDVI = (B8 - B4) / (B8 + B4)
```

---

# 🕒 Multi-Temporal Representation

The project uses two observations representing different time periods.

## Experiment 1 & Experiment 2

The input contains:

```text
Year 1
  7 channels
     +
Year 2
  7 channels
     =
14 channels
```

Therefore:

```text
Input shape = (14, 224, 224)
```

## Experiment 3 – Temporal Difference Representation

Experiment 3 extends the 14-channel representation by explicitly calculating the temporal difference:

```text
Delta = Year 2 - Year 1
```

Seven additional temporal-difference channels are created:

```text
Delta_B2
Delta_B3
Delta_B4
Delta_B8
Delta_NDVI
Delta_VV
Delta_VH
```

The final representation is:

```text
Year 1       → 7 channels
Year 2       → 7 channels
Year 2-Year1 → 7 channels
                  ↓
             21 channels
```

Therefore:

```text
Input shape = (21, 224, 224)
```

This representation is designed to help the network learn **what changed between the two observations**, rather than relying only on the appearance of either individual observation.

---

# 🛰️ Data Sources

## Sentinel-2

Sentinel-2 provides multispectral optical information.

ForestWatch uses:

- **B2 – Blue**
- **B3 – Green**
- **B4 – Red**
- **B8 – Near Infrared**

NDVI is derived from B8 and B4.

## Sentinel-1

Sentinel-1 provides Synthetic Aperture Radar information.

ForestWatch uses:

- **VV**
- **VH**

SAR provides complementary information about surface and vegetation structure and is useful because radar observations are less dependent on visible-light conditions than optical imagery.

---

# 🗺️ Study Area

The project focuses on selected forest regions of the **Western Ghats, Karnataka, India**.

The study region contains a mixture of forest, agricultural, and other land-cover types, making it suitable for investigating the distinction between genuine forest-cover changes and normal temporal variation.

---

# ⚙️ Data Preprocessing

The satellite data is prepared before being supplied to the deep-learning model.

The preprocessing pipeline includes:

- Radiometric/calibration-related preprocessing
- SAR speckle filtering
- Cloud masking for optical data
- Resampling
- Spatial co-registration/alignment
- NDVI calculation
- Multi-sensor feature construction
- Patch generation
- Training-only normalization statistics

The final satellite patches are stored as NumPy arrays (`.npy`).

---

# 🧩 Dataset Preparation

The processed satellite data is divided into spatial patches of:

```text
224 × 224 pixels
```

A stride of:

```text
112 pixels
```

is used during patch generation, producing overlapping patches.

The final dataset contains:

| Split | Number of Patches | Percentage |
|---|---:|---:|
| Training | 248 | 52.5% |
| Validation | 112 | 23.7% |
| Testing | 112 | 23.7% |
| **Total** | **472** | **100%** |

The split is spatially separated rather than being a simple random split, helping reduce the risk of spatial leakage between training and evaluation data.

### Normalization

Channel-wise normalization is performed using statistics calculated from the **training set only**.

This prevents information from the validation or test sets from being used to calculate normalization parameters.

---

# 🧠 Deep Learning Model

## Modified ResNet50

ForestWatch uses a **pre-trained ResNet50** architecture.

The original ImageNet ResNet50 accepts:

```text
3 channels
```

The project modifies the first convolutional layer to accept:

- **14 channels** for Experiments 1 and 2
- **21 channels** for Experiment 3

The remaining ResNet50 architecture is retained, and the final classification layer is modified for:

```text
2 classes
```

### Classes

```text
0 → Negative
1 → Positive
```

The model performs binary classification of satellite patches.

---

# ❓ Why ResNet50?

ResNet50 was selected because:

- It is a deep CNN with strong spatial feature extraction capability.
- Residual/skip connections make deep networks easier to optimize.
- ImageNet pre-trained weights provide useful transferable visual features.
- The model can be adapted to multi-channel satellite inputs.
- It provides a practical balance between model capacity and computational requirements for the available dataset.

The project currently uses ResNet50 consistently across the experiments so that the main comparison focuses on **input representation and training strategy**, rather than changing the backbone architecture.

---

# 🧪 Experiments

## Experiment 1 – 14-Channel Baseline

### Input

```text
Year 1 → 7 channels
Year 2 → 7 channels
Total  → 14 channels
```

This establishes the baseline performance of the multi-temporal, multi-sensor representation.

---

## Experiment 2 – 14-Channel ResNet50 with Regularization

Experiment 2 keeps the same 14-channel input but improves the training strategy.

### Augmentation

- Horizontal flip
- Vertical flip
- 90-degree rotation

### Regularization

```text
Weight decay = 0.001
```

Learning rates:

```text
Warmup      = 5e-5
Fine-tuning = 5e-6
```

This experiment investigates whether better regularization and augmentation improve generalization.

---

## Experiment 3 – 21-Channel Temporal ResNet50

Experiment 3 adds explicit temporal-difference information.

```text
Year 1
   +
Year 2
   +
Year 2 - Year 1
   ↓
21-channel input
```

The same augmentation and regularization strategy used in Experiment 2 is retained.

This experiment directly tests whether explicitly representing temporal change improves the classification results.

---

# 🔄 Overall Processing Pipeline

```text
Sentinel-1 + Sentinel-2
          ↓
      Preprocessing
          ↓
  7 Features per Year
          ↓
     Year 1 + Year 2
          ↓
     14-Channel Input
          ↓
   Experiment 1 / 2
          ↓
   Temporal Difference
     Year2 - Year1
          ↓
      21 Channels
          ↓
   Modified ResNet50
          ↓
   Binary Classification
          ↓
    Test Evaluation
```

---

# 📊 Experimental Results

All final metrics below are calculated on the **held-out test set of 112 patches**.

| Experiment | Input | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | 14-channel | 64.29% | 64.20% | 82.54% | 72.22% | 65.66% | 68.04% |
| Experiment 2 | 14-channel + regularization | 84.82% | 81.08% | 95.24% | 87.59% | 96.66% | 98.03% |
| Experiment 3 | 21-channel temporal difference | **91.07%** | **90.77%** | 93.65% | **92.19%** | 96.83% | 98.15% |

### Key Observation

Experiment 3 achieved the best overall test performance:

```text
Accuracy  = 91.07%
Precision = 90.77%
Recall    = 93.65%
F1-Score  = 92.19%
ROC-AUC   = 96.83%
PR-AUC    = 98.15%
```

Compared with the baseline, the 21-channel representation substantially improves the held-out test results.

The results support the hypothesis that explicitly representing temporal differences can provide useful information for distinguishing forest-cover changes from other variations in the current dataset.

> **Important:** The improvement should not be interpreted as proof that the temporal-difference channels alone caused the entire performance increase. A dedicated ablation/control experiment would be required to isolate their contribution from the effect of increasing the number of input channels.

---

# 📉 Confusion Matrices

### Baseline

```text
TN = 20
FP = 29
FN = 11
TP = 52
```

### Experiment 2

```text
TN = 35
FP = 14
FN = 3
TP = 60
```

### Experiment 3

```text
TN = 43
FP = 6
FN = 4
TP = 59
```

Experiment 3 produces substantially fewer false positives than the baseline while maintaining high recall.

---

# 📈 Validation and Training

The best validation F1-scores were:

| Experiment | Best Validation F1 | Best Epoch |
|---|---:|---:|
| Experiment 1 | 0.4500 | 1 |
| Experiment 2 | 0.5714 | 15 |
| Experiment 3 | 0.5769 | 15 |

Experiments 2 and 3 used early stopping after five epochs without validation improvement.

---

# 🔍 Evaluation Metrics

The project evaluates the models using:

### Accuracy

Measures the proportion of correctly classified samples.

### Precision

Measures how many predicted positive samples were actually positive.

### Recall

Measures how many actual positive samples were detected.

### F1-Score

Balances precision and recall and is particularly useful when both missed detections and false detections matter.

### ROC-AUC

Measures discrimination between the two classes across classification thresholds.

### PR-AUC

Measures the precision-recall trade-off and is useful for evaluating positive-class detection performance.

---

# 🧪 Validation and Data Leakage Checks

The project includes dataset sanity checks before evaluation.

Verified checks include:

- Test set contains exactly 112 patches.
- No duplicate test filenames.
- No duplicate test rows.
- No train/test filename overlap.
- No validation/test filename overlap.
- No train/validation filename overlap.
- Normalization statistics are calculated from training data only.
- Model checkpoint and training history are verified.

A coordinate-level leakage check could not be performed because the available metadata did not contain the expected coordinate columns. Therefore, the project does **not** claim that coordinate-level leakage was independently verified.

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep-learning framework and training pipeline |
| Torchvision | ResNet50 architecture and ImageNet weights |
| NumPy | Satellite patch arrays and temporal-difference calculations |
| Pandas | Metadata and dataset management |
| JSON | Channel normalization statistics |
| Matplotlib | Training curves, confusion matrices, and result visualization |
| Sentinel-1 | SAR data: VV and VH |
| Sentinel-2 | Optical data: B2, B3, B4 and B8 |
| QGIS | Geospatial visualization and inspection |
| Google Colab | GPU-based experimentation when required |
| Git | Version control |
| GitHub | Source-code hosting and collaboration |

---

# 📁 Project Structure

The implemented project is organized around data processing, model development, training, evaluation, and results:

```text
forestwatch-deforestration-detection/
│
├── data/
│   └── processed/
│       ├── patches/
│       ├── channel_stats.json
│       ├── experiment3_channel_stats.json
│       └── models/
│           ├── experiment2_best_resnet50.pth
│           ├── experiment2_training_history.csv
│           ├── experiment3_best_resnet50.pth
│           └── experiment3_training_history.csv
│
├── results/
│   └── metrics/
│       ├── baseline_14ch_metrics.csv
│       ├── experiment2_metrics.csv
│       ├── experiment3_metrics.csv
│       ├── experiment_comparison.csv
│       ├── baseline_14ch_confusion_matrix.png
│       └── ...
│
├── scripts/
│   ├── model_resnet50_14ch.py
│   ├── train_resnet50_14ch.py
│   ├── evaluate_resnet50_14ch.py
│   ├── calculate_channel_stats.py
│   ├── test_dataloader.py
│   ├── test_normalized_dataloader.py
│   └── test_training_pipeline.py
│
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

> The exact file structure may continue to evolve as the project is developed.

---

# ⚙️ Installation and Setup

## 1. Clone the Repository

```bash
git clone git@github-nmit:nmit-1nt23cs245/forestwatch-deforestration-detection.git
```

## 2. Navigate to the Project Directory

```bash
cd forestwatch-deforestration-detection
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The training pipeline requires the Python/PyTorch environment used by the project.

---

# 🚀 Running the Project

The implemented workflow follows these major stages:

### Step 1 – Prepare Satellite Data

Prepare Sentinel-1 and Sentinel-2 data for the selected study region and time periods.

### Step 2 – Preprocess the Data

Perform the required preprocessing, alignment, feature generation, and NDVI calculation.

### Step 3 – Generate Patches

Generate:

```text
224 × 224
```

satellite patches and prepare the corresponding metadata.

### Step 4 – Calculate Training Statistics

Calculate channel-wise normalization statistics using the training data only.

### Step 5 – Test the DataLoader

Verify:

- dataset sizes,
- input dimensions,
- labels,
- data types,
- normalization.

### Step 6 – Train the Models

Run the experiments using the modified ResNet50 pipeline.

### Step 7 – Evaluate the Models

Evaluate the best checkpoints on the held-out test set using:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
PR-AUC
```

### Step 8 – Compare Experiments

The experiment results are stored in:

```text
results/metrics/experiment_comparison.csv
```

---

# 📦 Model Checkpoints

The trained model checkpoints include:

```text
data/processed/models/experiment2_best_resnet50.pth
data/processed/models/experiment3_best_resnet50.pth
```

The corresponding training histories are stored as CSV files.

---

# 📊 Current Project Status

## Status: Model Development and Evaluation Completed

The current implementation includes:

- Multi-sensor Sentinel-1/Sentinel-2 feature representation
- NDVI generation
- Multi-temporal 14-channel input construction
- Training-only normalization
- Spatial patch generation
- Train/validation/test dataset preparation
- Modified 14-channel ResNet50
- Regularization and augmentation experiment
- 21-channel temporal-difference representation
- Modified 21-channel ResNet50
- Training and validation monitoring
- Early stopping
- Model checkpointing
- Held-out test evaluation
- Confusion matrix generation
- Experiment comparison
- Performance metrics including ROC-AUC and PR-AUC

The strongest current result is from **Experiment 3**, which achieved an F1-score of **92.19%** and accuracy of **91.07%** on the held-out test set.

---

# ⚠️ Limitations

The current results should be interpreted within the scope of the available dataset.

- The dataset contains 472 patches, so larger datasets are needed for stronger conclusions.
- The current evaluation is based on the selected study region and available test set.
- Generalization to other geographical regions has not yet been established.
- Seasonal and land-use variations can still create challenging cases.
- A dedicated ablation study is required to isolate the contribution of the temporal-difference channels from the increase in input dimensionality.
- The current project uses ResNet50 consistently rather than performing a complete architecture comparison.
- Coordinate-level leakage verification was limited by the available metadata fields.

---

# 🔮 Future Scope

Future work can include:

- Expanding the dataset to larger areas of the Western Ghats.
- Adding more temporal observations.
- Performing cross-region and cross-year validation.
- Conducting a dedicated temporal-difference ablation study.
- Comparing ResNet50 with other CNN and transformer architectures.
- Improving detection of small and fragmented forest-loss regions.
- Developing automated deforestation alerts.
- Generating GIS-based forest-loss maps.
- Estimating affected forest area.
- Extending the system toward forest degradation monitoring.
- Integrating additional environmental and satellite datasets.
- Deploying the model as a web-based monitoring system.

---

# 👥 Project Team

## ForestWatch – Engineering Major Project

| Name | Role |
|---|---|
| **SUDHARSHAN KS** | Team Member |
| **Deepak** | Team Member |
| **Shaikh Mohammed Faizan** | Team Member |
| **Shreyas S M** | Team Member |

---

# 👨‍🏫 Project Guide

**Ms. Archana M**  
**Designation:** Assistant Professor  
**Department:** Department of Computer Science and Engineering  
**Institution:** Nitte Meenakshi Institute of Technology, Bengaluru

---

# 📚 References

The project is based on research and resources related to:

- Satellite-based forest monitoring
- Remote sensing
- Multi-temporal change detection
- Sentinel-1 SAR
- Sentinel-2 multispectral imagery
- NDVI-based vegetation analysis
- Deep learning for remote sensing
- ResNet50 and transfer learning
- GIS-based forest monitoring

The complete research-paper bibliography is maintained in the project documentation/final report.

---

# 📌 Project Information

| Parameter | Details |
|---|---|
| **Project Name** | ForestWatch – Satellite Image Analysis for Deforestation Detection |
| **Project Type** | Engineering Major Project |
| **Domain** | Remote Sensing, Deep Learning, Computer Vision, Environmental Monitoring |
| **Study Area** | Western Ghats, Karnataka, India |
| **Primary Data Sources** | Sentinel-1 and Sentinel-2 |
| **Features** | B2, B3, B4, B8, NDVI, VV, VH |
| **Temporal Representation** | Year 1 + Year 2 + Year2-Year1 |
| **Final Input** | 21 channels |
| **Patch Size** | 224 × 224 |
| **Dataset Size** | 472 patches |
| **Deep Learning Model** | Modified ResNet50 |
| **Classes** | Binary classification |
| **Best Test Accuracy** | 91.07% |
| **Best Test F1-Score** | 92.19% |
| **Best Test ROC-AUC** | 96.83% |
| **Development Platform** | Python / PyTorch / Google Colab |
| **Version Control** | Git / GitHub |

---

# ⭐ Project Goal

> **ForestWatch aims to investigate a multi-sensor and multi-temporal deep-learning approach for identifying potential forest-cover loss, with explicit temporal-difference features designed to help distinguish meaningful changes from normal temporal vegetation variation.**
