# CSCI-4341-FInalProject
FInal Project for the Deep Learning for Medical Imaging CSCI - 4341
# Reimplementing CheXNet and EfficientNet for Pneumonia Detection on Chest X-ray Images

## Project Overview

This project reimplements and evaluates the CheXNet model (based on DenseNet-121) and EfficientNet for binary classification of pneumonia in chest X-ray images. The objective is to reproduce and improve upon CheXNet's performance using publicly available data and modern techniques like transfer learning and Grad-CAM visualization.

---

## Team Members

* Marcelo Cavazos
* Mark Rodriguez
* Joe Reyna

---

## Dataset

We use the **ChestX-ray14** dataset, which consists of:

* **112,120** frontal chest X-ray images
* Collected from **30,805** patients at the NIH Clinical Center
* 14 thoracic disease labels (focus: *Pneumonia*)
* Publicly accessible via Kaggle: [https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)

### Data Subset Used

As recommended, we selected a subset of **2,120 images** for computational feasibility. The subset was split as follows:

* **Training**: 70% (1,484 images)
* **Validation**: 15% (318 images)
* **Test**: 15% (318 images)

---

## Methods / Approach

1. **Reimplementation of CheXNet** using **DenseNet-121** trained with **binary cross-entropy loss**.
2. **Implementation of EfficientNet** for comparison.
3. **Transfer learning** using pretrained ImageNet weights.
4. **Training configuration**:

   * Optimizer: **Adam**
   * Learning rate: `1e-4`
   * Batch size: `16`
   * **Early stopping** to prevent overfitting

---

## Evaluation Plan

### Quantitative Metrics

* Accuracy
* Precision, Recall, F1-score
* AUC-ROC
* Confusion matrix to evaluate classification errors

### Qualitative Analysis

* **Grad-CAM visualizations** (optional but included if feasible)
* **Error analysis** for insights on misclassifications and model limitations

---

## Hypothesis

Reimplementing CheXNet and EfficientNet is expected to achieve **near radiologist-level performance (AUC > 0.85)** in pneumonia detection. Improvements in generalization and recall may be observed through:

* Data augmentation
* Use of EfficientNet architecture
* Grad-CAM visualization for better interpretability

---

## Getting Started

### Requirements

* Python 3.8+
* PyTorch
* torchvision
* matplotlib
* pandas
* scikit-learn
* PIL

### Setup

```bash
pip install -r requirements.txt
```

### Running the Project

```bash
python main.py
```

---

## Acknowledgements

* NIH Clinical Center for the ChestX-ray14 dataset
* Original CheXNet paper: [https://arxiv.org/abs/1711.05225](https://arxiv.org/abs/1711.05225)
* DenseNet and EfficientNet authors
* CSCI 4341: Deep Learning for Medical Imaging
