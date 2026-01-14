# Performance Benchmarking of Object Detection Models for Fetal Brain Ultrasound in Resource-Constrained Environments

**Authors:** Richard Huang, Advenia Tricahya Wiyono, Alvina Krisendi, Said Achmad
**Affiliation:** Computer Science Department, School of Computer Science, Bina Nusantara University, Jakarta, Indonesia

---

## Overview

This repository contains the official implementation for the research paper *"Performance Benchmarking of Object Detection Models for Fetal Brain Ultrasound in Resource-Constrained Environments: Accuracy, Efficiency, and Impact of Augmentation."*

The study evaluates the trade-off between detection performance and computational expense for detecting the Cavum Septum Pellucidum (CSP) and Lateral Ventricles (LV) in fetal ultrasound images. It compares single-stage (YOLO variants, SSD) and two-stage (Faster R-CNN) detectors across two experimental phases: raw data training and on-the-fly augmented training.

## Key Features

* **Models Evaluated:** YOLOv5, YOLOv8, YOLO11, YOLO12, SSD (VGG16, EfficientNetB2), and Faster R-CNN (MobileNetV3 Small/Large).
* **Methodology:** A comparative analysis focusing on resource-constrained environments, utilizing both low-parameter and high-parameter architectural variants.
* **Data Strategy:** Patient-level stratified splitting to prevent data leakage, ensuring robust clinical validation.

---

## Repository Structure

* `YOLO_Fetal_Head_Object_Detection_Code.ipynb`: The primary control script. Handles data cleaning, patient-ID extraction, dataset splitting (Train/Val/Test), YAML configuration generation, and YOLO model training.
* `SSD_VGG16.ipynb`: Implementation of the Single Shot MultiBox Detector with a VGG16 backbone.
* `SSD_EfficientnetB2.ipynb`: Implementation of SSD with a lightweight EfficientNetB2 backbone.
* `Faster_RCNN_MobileNetV3_Small.ipynb`: Faster R-CNN implementation optimized for low-resource settings.
* `Faster_RCNN_MobileNetv3_Large.ipynb`: Faster R-CNN implementation using the larger MobileNetV3 backbone for higher capacity.

---

## Usage Instructions

**CRITICAL: Execution Order is Mandatory**

This repository relies on a specific data preparation pipeline. You must execute the notebooks in the following order to ensure all dependencies and dataset splits are correctly initialized.

### Step 1: Data Preparation and YOLO Training
**Run:** `YOLO_Fetal_Head_Object_Detection_Code.ipynb`

* **Function:** This notebook contains the `split_dataset()` function, which performs the stratified split based on patient IDs. It generates the `FINAL_YOLO_SPLIT` directory and the `dataset.yaml` file.
* **Dependency:** All subsequent models (SSD, Faster R-CNN) depend on the file structure and configuration files created in this step. Attempting to run other notebooks first will result in file path errors.

### Step 2: Benchmarking Other Architectures
Once Step 1 is complete, you may run the remaining notebooks in any order to reproduce the comparative results:

* `SSD_VGG16.ipynb`
* `SSD_EfficientnetB2.ipynb`
* `Faster_RCNN_MobileNetV3_Small.ipynb`
* `Faster_RCNN_MobileNetv3_Large.ipynb`

---

## Methodology

### Data Preprocessing
The dataset is partitioned into Training (70%), Validation (15%), and Testing (15%) sets. Crucially, grouping is performed based on unique patient identifiers to ensure that images from the same subject do not appear in both training and testing sets, thereby preventing data leakage.

### Experimental Design
The research compares models in two groups:
1.  **Low-Parameter Group:** Designed for edge devices (e.g., YOLOv8s, YOLO11s, SSD-EfficientNetB2, Faster R-CNN MobileNetV3-Small).
2.  **High-Parameter Group:** Designed for higher capacity (e.g., YOLOv8m, YOLO12m, SSD-VGG16, Faster R-CNN MobileNetV3-Large).

Experiments are conducted using two data flows:
* **Raw:** Default training settings.
* **Tuned:** Integration of on-the-fly augmentations including rotation, shear, perspective, and translation to simulate operator variability.

---

## Results Summary

The study demonstrates that higher model complexity does not strictly correlate with better performance in fetal ultrasound biometry.

* **Low-Parameter Optimal:** YOLOv8s achieved the best balance of accuracy and efficiency (mAP@50: 0.841, 28.4 GFLOPs).
* **High-Parameter Optimal:** YOLO12m achieved the highest overall accuracy (mAP@50: 0.85) but incurred significantly higher computational costs.
* **Conclusion:** YOLO-based architectures generally demonstrated superior suitability for resource-constrained environments compared to SSD and Faster R-CNN.

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

If you utilize this code or dataset in your research, please cite the following paper:

Huang, R., Wiyono, A. T., Krisendi, A., & Achmad, S. (2026). *Performance Benchmarking of Object Detection Models for Fetal Brain Ultrasound in Resource-Constrained Environments: Accuracy, Efficiency, and Impact of Augmentation*. Bina Nusantara University.
