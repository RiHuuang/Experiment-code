# Performance Benchmarking of Object Detection Models for Fetal Brain Ultrasound in Resource-Constrained Environments

**Authors:** Richard Huang, Advenia Tricahya Wiyono, Alvina Krisendi, Said Achmad
**Affiliation:** Computer Science Department, School of Computer Science, Bina Nusantara University, Jakarta, Indonesia

---

## Overview

This repository contains the official implementation for the research paper *"Performance Benchmarking of Object Detection Models for Fetal Brain Ultrasound in Resource-Constrained Environments: Accuracy, Efficiency, and Impact of Augmentation."*

The study evaluates the trade-off between detection performance and computational expense for detecting the Cavum Septum Pellucidum (CSP) and Lateral Ventricles (LV) in fetal ultrasound images. It compares single-stage (YOLO variants, SSD) and two-stage (Faster R-CNN) detectors across two experimental phases: raw data training and on-the-fly augmented training.

---

## 📂 Repository Structure

* `YOLO_Fetal_Head_Object_Detection_Code.ipynb`: **(Control Script)** Handles data cleaning, patient-ID extraction, dataset splitting (Train/Val/Test), YAML configuration generation, and YOLO model training.
* `SSD_VGG16.ipynb`: Implementation of SSD with a VGG16 backbone.
* `SSD_EfficientnetB2.ipynb`: Implementation of SSD with a lightweight EfficientNetB2 backbone.
* `Faster_RCNN_MobileNetV3_Small.ipynb`: Faster R-CNN optimized for low-resource settings.
* `Faster_RCNN_MobileNetv3_Large.ipynb`: Faster R-CNN with a high-capacity backbone.

---

## 🧬 Data Acquisition & Preparation

### 1. Dataset Source
This study utilizes the **"Large-scale annotation dataset for fetal head biometry in ultrasound images"** provided by **Alzubaidi et al. (2023)**.
* **Original Paper:** *Data in Brief*, Vol. 51, 109708.
* **Content:** 3,832 high-resolution ultrasound images with expert annotations for Brain, CSP, and LV.

### 2. Preparation Workflow
To reproduce our data pipeline, follow these specific preprocessing steps:

1.  **Download:** Obtain the dataset in **YOLO format** from the official repository (https://doi.org/10.5281/zenodo.8265464).
2.  **Extract & Combine:**
    * Extract the downloaded files.
    * Combine all image and label files into a single master directory named **`YOLO Dataset Full`**.
3.  **Processing:**
    * The `YOLO_Fetal_Head_Object_Detection_Code.ipynb` notebook takes this `YOLO Dataset Full` directory as input.
    * It processes the raw data into standardized images and labels.
    * It then executes the **stratified splitting** (Train/Val/Test) based on Patient ID to ensure no data leakage occurs.

We provided all the data folders here **[Download Here](https://drive.google.com/drive/folders/1xbvf5IdsfUsJNqOg17mqob-4sVEAmb2d?usp=sharing)**
---

## 🚀 Usage Instructions

**CRITICAL: Execution Order is Mandatory**

### Step 1: Data Preparation and YOLO Training
**Run:** `YOLO_Fetal_Head_Object_Detection_Code.ipynb`

* **Input:** Reads from `Dataset/Yolo Dataset Full` (as described above).
* **Action:** Performs patient-aware splitting and generates the `dataset.yaml` configuration file.
* **Output:** Creates the `FINAL_YOLO_SPLIT` directory.
* **Why this is first:** All subsequent models (SSD, Faster R-CNN) depend on the file structure and `dataset.yaml` created here.

### Step 2: Benchmarking Other Architectures
Once Step 1 is complete, run the remaining notebooks in any order:
* `SSD_VGG16.ipynb`
* `SSD_EfficientnetB2.ipynb`
* `Faster_RCNN_MobileNetV3_Small.ipynb`
* `Faster_RCNN_MobileNetv3_Large.ipynb`

---

## Methodology

### Data Preprocessing
The dataset is partitioned into **Training (70%)**, **Validation (15%)**, and **Testing (15%)** sets. Crucially, grouping is performed based on unique **Patient IDs** to ensure that images from the same subject do not bleed across splits, thereby preventing data leakage and ensuring clinical validity.

### Experimental Design
The research compares models in two groups:
1.  **Low-Parameter Group:** Designed for edge devices (e.g., YOLOv8s, YOLO11s, SSD-EfficientNetB2, Faster R-CNN MobileNetV3-Small).
2.  **High-Parameter Group:** Designed for higher capacity (e.g., YOLOv8m, YOLO12m, SSD-VGG16, Faster R-CNN MobileNetV3-Large).

Experiments are conducted using two data flows:
* **Raw:** Default training settings.
* **Tuned:** On-the-fly augmentations (rotation, shear, perspective, translation) to simulate operator variability.

---

## Requirements

* Python 3.8+
* PyTorch
* Torchvision
* Ultralytics
* Albumentations
* Pandas
* NumPy
* OpenCV
* Matplotlib
* Seaborn

## Citation
```bibtex
@article{Alzubaidi2023Dataset,
  title={Large-scale annotation dataset for fetal head biometry in ultrasound images},
  author={Alzubaidi, Mahmood and Agus, Marco and Makhlouf, Michel and Anver, Fatima and Alyafei, Khalid and Househ, Mowafa},
  journal={Data in Brief},
  volume={51},
  pages={109708},
  year={2023},
  publisher={Elsevier}
}
