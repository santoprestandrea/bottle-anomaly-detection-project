# Industrial Anomaly Detection — MVTec Bottle Category
 
A Computer Vision pipeline for unsupervised anomaly detection on industrial images,
built as the final project for the *Introduction to Computer Vision* course at Epicode Institute of Technology.
 
---
 
## Project Overview
 
This project addresses the problem of automated quality control in manufacturing.
The goal is to detect visual defects (fractures, contamination) in bottle images
without using any defective samples during training — a setting known as **one-class learning**.
 
The pipeline is a simplified implementation of **PatchCore** (Roth et al., 2022):
patch-level features are extracted from normal training images using a pre-trained
ResNet18 backbone, stored in a memory bank, and compared at test time using
k-Nearest Neighbors. High distance from the memory bank signals an anomaly.
 
---
 
## Pipeline
 
```
Input Image
 │
 ▼
Preprocessing
(resize, normalization, augmentation)
 │
 ▼
Feature Extraction
(ResNet18 truncated at layer2 → 784 patch descriptors × 128 features)
 │
 ▼
Anomaly Scoring
(k-NN on memory bank of normal patches → patch map 28×28)
 │
 ▼
Post-processing
(Gaussian smoothing → bilinear upsampling → thresholding → morphological ops)
 │
 ▼
Pixel-level Anomaly Mask (224×224)
```
 
---
 
## Dataset
 
**MVTec Anomaly Detection** — `bottle` category (Bergmann et al., 2019).
 
| Split | Content | Count |
|-------|---------|-------|
| Train | Normal images only | ~209 |
| Test — normal | Defect-free bottles | ~20 |
| Test — defective | `broken_large`, `broken_small`, `contamination` | ~63 |
 
Download the dataset from the [official MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad).
 
---
 
## Results
 
| Metric | Value |
|--------|-------|
| Image-level AUROC | **0.9937** |
| Pixel-level IoU | **0.2779** |
| Pixel-level Dice | **0.4349** |
 
The image-level AUROC of 0.9937 indicates near-perfect separation between normal
and defective images. Pixel-level scores reflect the known limitation of patch-based
methods at coarse resolution (28×28): the system correctly localizes defects but
tends to underestimate their spatial extent.
