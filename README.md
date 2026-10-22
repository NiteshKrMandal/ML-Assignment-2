# ML Assignment 2 — Olivetti Faces

**Student:** Nitesh Kumar Mandal  
**Roll number:** M26CSA005

## Overview

This project studies decision-tree classification, bootstrap aggregation (bagging), multiclass SAMME boosting, and measured bias–variance behavior on the Olivetti Faces dataset.

The dataset contains 400 grayscale face images from 40 identities. Each image is represented as a 64 × 64 image (4096 features). The split is stratified by identity with 6 training, 2 validation, and 2 test images per identity.

The masked test input has the central region `rows 24:39, columns 24:39` set to zero.

## Repository contents

- `ml-assignment-2.ipynb` — analysis and experiments for Questions 1–3.
- `olivetti_split.npz` — saved split indices. Add this file from the completed notebook run; do not regenerate it if you need exact reproduction of the reported results.
- `config.json` — key experiment settings.
- `results.json` — summary of the final reported measurements.
- `requirements.txt` — Python dependencies.
- `figures/` — place exported final plots here.

## Dataset

The notebook uses `sklearn.datasets.fetch_olivetti_faces()`.

## Reproduce

1. Open `ml-assignment-2.ipynb` in Jupyter or VS Code.
2. Run Kernal.
3. Restart and Run All

## Final reported results

- Selected single decision tree: 46.25% clean test accuracy.
- Bagging (100 trees): 91.25% clean test accuracy; 72.50% masked test accuracy.
- SAMME boosting (depth 4, 50 rounds): 35.00% clean test accuracy; 17.50% masked test accuracy.
- Q3 measured bias-squared / variance:
  - Single tree: 0.4243 / 0.6674
  - Bagging: 0.1652 / 0.2748
  - Boosting: 0.7676 / 0.7324
