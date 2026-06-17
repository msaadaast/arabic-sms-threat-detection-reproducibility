# Splits

This folder contains the fixed split files used to reproduce the reported experiments.

## Files

`split_holdout10_cv5_seed42.npz`  
Fixed split file for the primary Arabic SMS dataset. It contains the development indices, untouched holdout indices, and 5-fold cross-validation indices used across SMS text-classification models.

`ext_splits_holdout10_cv5_seed42.npz`  
Fixed split file for the external Arabic short-text dataset used in the cross-domain methodological robustness experiment.

`url_shared_splits_holdout10_cv5.npz`  
Fixed split file for the smishing-oriented URL dataset. It contains the rare-class stratified development/holdout split and 5-fold cross-validation indices used across the URL-analysis models.

## Notes

All split files were generated using a fixed random seed of 42. The purpose of storing these split files is to ensure that all model families are evaluated on identical development, cross-validation, and holdout partitions.

The primary SMS split file is shared across the classical ML, 1D-CNN, DistilBERT, AraBERT, and MARBERT experiments.

The URL split file is shared across the heuristic engine, Linear SVM, XGBoost, and Char-CNN URL-analysis experiments.

The external split file is used only for the independent Arabic short-text methodological robustness experiment.

## Required placement

Place these files under the repository-level `splits/` folder:

```text
splits/
  splits_holdout10_cv5_seed42.npz
  ext_splits_holdout10_cv5_seed42.npz
  url_shared_splits_holdout10_cv5.npz
  README.md
```

## Reproducibility purpose

These split files preserve the exact development, holdout, and cross-validation partitions used in the manuscript. They are included to prevent accidental changes in model results caused by random re-splitting of the datasets.
