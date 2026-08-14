# ForestWatch – Satellite Image Analysis for Deforestation Detection

> **An AI-based multi-temporal satellite image analysis system for detecting forest-cover changes and potential deforestation using Sentinel-1 SAR, Sentinel-2 RGB, NDVI, and ResNet50.**

---

## 📌 Project Overview

**ForestWatch** is a satellite-based deforestation detection system developed to monitor changes in forest cover using **multi-temporal satellite imagery and deep learning**.

The system integrates:

* **Sentinel-1 SAR** data
* **Sentinel-2 RGB** imagery
* **NDVI (Normalized Difference Vegetation Index)**

to analyze forest regions across different time periods.

ForestWatch uses a **pre-trained ResNet50 convolutional neural network** for deep feature extraction and classification. The classified results from different time periods are then compared to identify changes in forest cover and potential deforestation regions.

The project focuses on selected forest regions of the **Western Ghats, Karnataka, India**, where seasonal vegetation changes, agricultural activities, cloud cover, and heterogeneous terrain make automated forest monitoring challenging.

---

## 🎯 Problem Statement

Traditional forest monitoring methods often require significant manual effort and are difficult to scale across large geographical regions.

Satellite-based monitoring provides an efficient alternative, but several challenges affect reliable deforestation detection:

* Cloud cover can affect optical satellite imagery.
* Seasonal vegetation variations can produce false deforestation signals.
* Agricultural crop cycles can resemble forest-cover changes.
* A single satellite data source may not provide sufficient information.
* Small or fragmented forest-loss regions can be difficult to identify.
* Different land-cover types can produce similar visual and spectral patterns.

Therefore, ForestWatch aims to develop a **multi-sensor and multi-temporal satellite image analysis system** that combines SAR, RGB, and NDVI information with deep learning to improve forest-cover change detection.

---

## 💡 Proposed Solution

ForestWatch combines complementary information from multiple satellite-derived data sources.

### Sentinel-1 SAR

Provides radar-based information about surface and structural characteristics and can complement optical imagery under cloudy conditions.

### Sentinel-2 RGB

Provides optical information about vegetation, land cover, and spatial characteristics of the study region.

### NDVI

Provides vegetation-related information and helps identify changes in vegetation density.

### ResNet50

A pre-trained **ResNet50 CNN** is used for deep feature extraction from satellite image patches. The extracted features are used for **forest/non-forest classification**.

### Multi-Temporal Change Detection

The classified results from two different time periods are compared to identify changes in forest cover.

For example:

**Time 1 (2022) → Time 2 (2024) → Forest-cover change → Potential deforestation**

---

## 🎯 Objectives

1. **To integrate Sentinel-1 SAR and Sentinel-2 optical satellite data for multi-temporal forest monitoring.**

2. **To develop a ResNet50-based approach for feature extraction and forest-cover change detection.**

3. **To evaluate the proposed system using Accuracy, Precision, Recall, and F1-Score.**

---

## 🛰️ Data Sources

### Sentinel-1

Sentinel-1 provides **Synthetic Aperture Radar (SAR)** imagery.

SAR data is used to obtain radar-based surface and structural information and provides complementary information to optical satellite imagery.

### Sentinel-2

Sentinel-2 provides multispectral optical satellite imagery.

ForestWatch uses:

* **B2 – Blue**
* **B3 – Green**
* **B4 – Red**
* **NIR band** for vegetation analysis

RGB imagery is generated using the selected visible bands.

### NDVI

NDVI is derived from Sentinel-2 Near-Infrared and Red bands.

The NDVI equation is:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

NDVI is used to represent vegetation density and support vegetation-change analysis.

---

## 🗺️ Study Area

The project focuses on selected forest regions of the **Western Ghats in Karnataka, India**.

The region is selected because of its:

* Dense forest cover
* High biodiversity
* Ecological importance
* Mixed forest and agricultural landscapes
* Seasonal vegetation variations
* Susceptibility to land-use changes and deforestation

The same selected **Region of Interest (ROI)** is analyzed across multiple time periods to enable multi-temporal comparison.

---

# 🏗️ System Architecture

ForestWatch follows a multi-layer architecture consisting of data acquisition, preprocessing, feature extraction, change detection, data management, data access, and application components.

The major architectural components are:

1. **Data Acquisition**
2. **Preprocessing & Data Preparation**
3. **Feature Extraction & Data Fusion**
4. **Change Detection & Classification**
5. **Results & Output**
6. **Data Management & Storage**
7. **Data Access Layer**
8. **Application Layer**
9. **Technology Stack**

### Architecture Diagram

![ForestWatch System Architecture](docs/architecture/forestwatch_architecture.png)

---

# 🔄 System Flowchart

The ForestWatch processing workflow follows the sequence:

```text
Data Acquisition
       ↓
Preprocessing
       ↓
Multi-Sensor Data Fusion
       ↓
Dataset Preparation
       ↓
ResNet50 Feature Extraction
       ↓
Forest / Non-Forest Classification
       ↓
Multi-Temporal Change Detection
       ↓
Output Generation
       ↓
Performance Evaluation
```

### Detailed Flowchart

![ForestWatch System Flowchart](docs/architecture/forestwatch_flowchart.png)

---

# ⚙️ Methodology

## 1. Data Acquisition

Satellite data is collected for the selected Region of Interest using **Google Earth Engine / Copernicus satellite datasets**.

The system uses:

```text
Sentinel-1 SAR
      +
Sentinel-2 RGB
      +
Sentinel-2 NDVI
```

Data is collected for multiple time periods, such as:

```text
Time 1 → 2022
Time 2 → 2024
```

---

## 2. Preprocessing & Data Preparation

Different preprocessing operations are applied to the satellite data.

### SAR Preprocessing

* Radiometric calibration
* Noise reduction
* Speckle filtering
* Terrain correction
* Resizing
* Normalization

### RGB Preprocessing

* Cloud removal
* Atmospheric correction
* Image alignment
* Band selection
* Resizing
* Normalization

### NDVI Processing

* Cloud removal
* NDVI calculation
* Resizing
* Normalization

The processed data is spatially aligned so that corresponding regions from different sensors and time periods can be compared.

---

## 3. Multi-Sensor Data Fusion

ForestWatch combines information from:

```text
Sentinel-1 SAR
       +
Sentinel-2 RGB
       +
Sentinel-2 NDVI
```

The complementary information from these sources is combined to form a multi-sensor feature representation.

The fusion approach is designed to provide more information about forest structure and vegetation than relying on a single data source.

---

## 4. Dataset Preparation

The large GeoTIFF satellite images are divided into smaller image patches suitable for deep learning.

The target patch size used in the project is:

```text
224 × 224 pixels
```

The dataset preparation process consists of:

1. Collecting multi-temporal satellite imagery.
2. Generating NDVI from Sentinel-2 data.
3. Preprocessing and normalizing the satellite images.
4. Spatially aligning images from different sensors and time periods.
5. Generating corresponding image patches.
6. Combining SAR, RGB, and NDVI information.
7. Assigning appropriate labels.
8. Dividing the dataset into training, validation, and testing sets.

---

## 5. Feature Extraction Using ResNet50

ForestWatch uses a **pre-trained ResNet50 CNN** for deep feature extraction.

ResNet50 uses residual connections that enable effective training of deep convolutional neural networks.

The model is used to learn meaningful spatial representations from satellite image patches.

### Why ResNet50?

* Deep CNN architecture
* Effective spatial feature extraction
* Residual learning
* Supports transfer learning
* Suitable for image classification
* Capable of learning complex visual patterns

The extracted features are used for **forest/non-forest classification**.

---

## 6. Forest / Non-Forest Classification

The extracted deep features are used to distinguish between:

```text
Forest
   ↓
Non-Forest
```

This classification provides a basis for comparing forest-cover status between different time periods.

---

## 7. Multi-Temporal Change Detection

The classified outputs for different time periods are compared.

For example:

```text
             TIME 1
             2022
               ↓
       Forest / Non-Forest
               ↓
          Comparison
               ↑
       Forest / Non-Forest
               ↑
             TIME 2
             2024
```

A region classified as forest in the earlier period and non-forest in the later period can be identified as a **potential forest-loss region**.

The system is designed to distinguish genuine forest-cover changes from changes caused by seasonal vegetation variation and other land-cover patterns.

---

# 📊 Dataset Preparation

The raw satellite GeoTIFF files are not directly used as the final deep-learning dataset.

The overall dataset pipeline is:

```text
Raw Satellite GeoTIFF
        ↓
Preprocessing
        ↓
Spatial Alignment
        ↓
SAR + RGB + NDVI
        ↓
Patch Generation
        ↓
224 × 224 Patches
        ↓
Label Assignment
        ↓
Training / Validation / Testing
```

The dataset is prepared to support multi-sensor and multi-temporal analysis.

---

# 🧠 Deep Learning Model

## ResNet50

ResNet50 is a 50-layer deep convolutional neural network based on residual learning.

In ForestWatch, a **pre-trained ResNet50** is used for deep feature extraction from satellite image patches.

The extracted representations support forest/non-forest classification before performing multi-temporal change detection.

The final model evaluation will use:

* Accuracy
* Precision
* Recall
* F1-Score

---

# 📈 Results

## Preliminary Multi-Temporal Analysis

A preliminary multi-modal change analysis was performed using **Chikkamagaluru satellite imagery from 2022 and 2024**.

The analysis used:

* Sentinel-1 SAR
* Sentinel-2 RGB
* NDVI

The three data sources were compared to identify regions showing significant changes between the two observation periods.

The preliminary analysis produced candidate change regions that require further validation using the trained ResNet50 model and appropriate labelled data before being considered confirmed deforestation.

### Current observations

The preliminary analysis demonstrates that:

* RGB imagery provides visual information about land-cover changes.
* NDVI helps identify vegetation-related changes.
* SAR provides complementary surface information.
* Combining multiple data sources can reduce dependence on a single indicator.
* Multi-temporal comparison provides a basis for identifying potential forest-cover loss.

### Model Performance

The final model performance will be reported after completion of training and testing.

| Metric    |           Score |
| --------- | --------------: |
| Accuracy  | To be evaluated |
| Precision | To be evaluated |
| Recall    | To be evaluated |
| F1-Score  | To be evaluated |

The **F1-Score** is particularly important because the system needs to balance the detection of actual forest-loss regions with the reduction of false detections.

---

# 🖼️ Results Visualization

ForestWatch generates visual outputs for analyzing satellite imagery and forest-cover changes.

### RGB Comparison

Shows the visual difference between satellite observations from different time periods.

### NDVI Comparison

Shows vegetation-related changes between the observation periods.

### SAR Comparison

Shows changes in radar-based surface characteristics.

### Deforestation Detection Map

Highlights potential forest-cover loss regions detected through multi-temporal analysis.

### Area Statistics

The final system is designed to provide statistics describing detected forest loss and forest gain.

---

# 📤 Output Generation

The proposed ForestWatch system generates the following outputs:

### Deforestation Map

```text
2022 vs 2024
```

showing potential forest-loss regions.

### Area Statistics

Provides information about detected forest loss and gain.

### Detection Report

Summarizes the detected changes and model results.

### Export Results

Results can be exported in formats such as:

```text
GeoTIFF
CSV
```

### Visualization Outputs

Maps and image-based visualizations are generated for interpretation and analysis.

---

# 🛠️ Technology Stack

| Technology          | Purpose                                   |
| ------------------- | ----------------------------------------- |
| Python              | Main programming language                 |
| Google Earth Engine | Satellite data acquisition and processing |
| Google Colab        | Development and model experimentation     |
| Sentinel-1          | SAR satellite data                        |
| Sentinel-2          | Optical satellite data                    |
| NDVI                | Vegetation analysis                       |
| TensorFlow / Keras  | Deep learning                             |
| ResNet50            | Feature extraction and classification     |
| Rasterio            | GeoTIFF processing                        |
| OpenCV              | Image processing                          |
| NumPy               | Numerical computation                     |
| SciPy               | Scientific and image processing           |
| Matplotlib          | Visualization                             |
| QGIS                | Geospatial analysis and visualization     |
| Git                 | Version control                           |
| GitHub              | Source-code management and collaboration  |

---

# 📁 Project Structure

```text
forestwatch-deforestration-detection/
│
├── data/
│   ├── raw/
│   │   ├── sentinel1/
│   │   └── sentinel2/
│   │
│   ├── processed/
│   │   ├── rgb/
│   │   ├── sar/
│   │   └── ndvi/
│   │
│   └── patches/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── dataset_generation.ipynb
│   └── resnet50_training.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── dataset/
│   ├── model/
│   ├── change_detection/
│   └── visualization/
│
├── results/
│   ├── rgb/
│   ├── sar/
│   ├── ndvi/
│   └── change_detection/
│
├── models/
│
├── docs/
│   ├── architecture/
│   └── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The project structure may be updated as implementation progresses.

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

## 4. Prepare the Dataset

Obtain the required Sentinel-1 and Sentinel-2 imagery for the selected Region of Interest.

Large satellite datasets should be stored locally or in an appropriate external/cloud storage location rather than directly inside the Git repository.

## 5. Run the Notebooks

The preprocessing, dataset generation, and model training notebooks can be executed using:

* Google Colab
* Jupyter Notebook

---

# 🚀 How to Run

### Step 1 – Acquire Satellite Data

Collect Sentinel-1 and Sentinel-2 imagery for the selected Region of Interest using Google Earth Engine.

### Step 2 – Preprocess Data

Run the preprocessing pipeline to prepare:

```text
SAR
RGB
NDVI
```

data.

### Step 3 – Generate Dataset

Generate spatially aligned multi-temporal image patches of size:

```text
224 × 224
```

and assign the corresponding labels.

### Step 4 – Train / Run ResNet50

Use the prepared dataset with the ResNet50-based feature extraction and classification pipeline.

### Step 5 – Perform Change Detection

Compare the classified outputs from different time periods.

### Step 6 – Evaluate the Model

Calculate:

```text
Accuracy
Precision
Recall
F1-Score
```

### Step 7 – Generate Results

Generate:

* Deforestation maps
* Area statistics
* Detection reports
* Visualization outputs
* GeoTIFF / CSV results

---

# 👥 Project Team

## ForestWatch – Engineering Major Project

| Name                    | Role        |
|-------------------------|-------------|
| **SUDHARSHAN KS**       | Team Member |
| **Deepak**              | Team Member |
| **Shaikh Mohammed Faizan** | Team Member |
| **Shreyas S M**          | Team Member |

> Replace the placeholders with the complete names of the remaining team members.

---

# 👨‍🏫 Project Guide

**Project Guide:**
**Ms. Archana M**
**Designation:** 
Assistant Professor
**Department:**
Department of Computer Science and Engineering

**Institution:**
Nitte Meenakshi Institute of Technology, Bengaluru

---

# 🔮 Future Scope

The ForestWatch system can be extended in several ways:

* Expand monitoring to larger regions of the Western Ghats.
* Support continuous or near-real-time forest monitoring.
* Develop automated deforestation alerts.
* Deploy the system as a web-based monitoring platform.
* Integrate additional satellite and environmental datasets.
* Improve detection of small-scale and fragmented forest loss.
* Explore advanced CNN and transformer-based architectures.
* Integrate interactive GIS-based visualization.
* Extend the system to monitor forest degradation.
* Extend the system to identify wildfire-related forest damage.
* Support monitoring of other land-use changes.

---

# 📚 References

The project is based on research and resources related to:

* Satellite-based forest monitoring
* Remote sensing
* Multi-temporal change detection
* SAR-based forest monitoring
* Sentinel-1 and Sentinel-2 satellite imagery
* NDVI-based vegetation analysis
* Deep learning for remote sensing
* ResNet50 and transfer learning
* GIS-based forest monitoring

A complete list of research papers and references is maintained in the project documentation and final project report.

---

# 📄 Project Information

| Parameter                | Details                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| **Project Name**         | ForestWatch – Satellite Image Analysis for Deforestation Detection       |
| **Project Type**         | Engineering Major Project                                                |
| **Domain**               | Remote Sensing, Deep Learning, Computer Vision, Environmental Monitoring |
| **Study Area**           | Western Ghats, Karnataka, India                                          |
| **Primary Data Sources** | Sentinel-1 and Sentinel-2                                                |
| **Input Data**           | SAR, RGB, NDVI                                                           |
| **Deep Learning Model**  | ResNet50                                                                 |
| **Analysis**             | Multi-Temporal Forest-Cover Change Detection                             |
| **Evaluation Metrics**   | Accuracy, Precision, Recall, F1-Score                                    |
| **Development Platform** | Python / Google Colab                                                    |
| **Version Control**      | Git / GitHub                                                             |

---

# 📌 Project Status

**Current Stage:** Prototype / Proof of Concept

The current project development includes satellite data acquisition, preprocessing, multi-sensor data preparation, multi-temporal image analysis, and development of the ResNet50-based deforestation detection pipeline.

The next stages include:

* Complete dataset labeling
* Dataset generation
* ResNet50 training
* Model validation and testing
* Performance evaluation
* Deforestation map generation
* Area-based analysis
* Final result validation

---

## ⭐ Project Goal

> **ForestWatch aims to provide an automated, scalable, and multi-sensor approach for monitoring forest-cover changes and identifying potential deforestation using satellite imagery and deep learning.**

---
