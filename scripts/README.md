# Scripts

This folder contains the experimental scripts used to reproduce the main model training, evaluation, and leakage-checking procedures reported in the paper.

## Important path configuration note

The scripts currently use absolute paths corresponding to the original experimental environment. Before running any script, update `DATA_PATH` and `OUTPUT_DIR` at the top of each script to match your local repository structure.

Recommended local structure:

```text
project_root/
  data/
    ARA_SMS_Dataset_Final.csv
    Smishing_Dataset_Final.csv
  scripts/
  outputs/
```

## Recommended path settings

| Script type | DATA_PATH | OUTPUT_DIR |
|---|---|---|
| SMS text-classification scripts | `data/ARA_SMS_Dataset_Final.csv` | `outputs/` |
| URL-analysis scripts | `data/Smishing_Dataset_Final.csv` | `outputs/smishing_results/` |
| Duplicate / leakage-check script | `data/ARA_SMS_Dataset_Final.csv` | `outputs/` |

## Script list

`01_check_sms_duplicates_leakage.py`  
Performs near-duplicate and post-split leakage analysis for the Arabic SMS dataset using character-level TF-IDF cosine similarity.

`02_train_sms_ml_models.py`  
Trains and evaluates classical machine-learning SMS classifiers using TF-IDF features.

`03_train_sms_1d_cnn.py`  
Trains and evaluates the 1D-CNN model for Arabic SMS spam classification.

`04_train_sms_distilbert.py`  
Fine-tunes and evaluates multilingual DistilBERT for Arabic SMS spam classification.

`05_train_sms_arabert.py`  
Fine-tunes and evaluates AraBERT for Arabic SMS spam classification.

`06_train_sms_marbert.py`  
Fine-tunes and evaluates MARBERT for Arabic SMS spam classification.

`07_run_url_heuristic_engine.py`  
Runs the interpretable URL heuristic engine, performs threshold evaluation, applies the selected threshold, and generates the shared URL split file.

`08_train_url_svm.py`  
Trains and evaluates the character-level TF-IDF + Linear SVM URL classifier.

`09_train_url_xgboost.py`  
Trains and evaluates the engineered-feature XGBoost URL classifier.

`10_train_url_char_cnn.py`  
Trains and evaluates the character-level CNN URL classifier.

## Recommended execution order

```bash
python scripts/02_train_sms_ml_models.py
python scripts/03_train_sms_1d_cnn.py
python scripts/04_train_sms_distilbert.py
python scripts/05_train_sms_arabert.py
python scripts/06_train_sms_marbert.py

python scripts/01_check_sms_duplicates_leakage.py

python scripts/07_run_url_heuristic_engine.py
python scripts/08_train_url_svm.py
python scripts/09_train_url_xgboost.py
python scripts/10_train_url_char_cnn.py
```

Run `07_run_url_heuristic_engine.py` before scripts `08_train_url_svm.py`, `09_train_url_xgboost.py`, and `10_train_url_char_cnn.py`, because the heuristic script creates the shared URL split file used by the learned URL models.

The transformer scripts require GPU acceleration for practical runtime. They can run on CPU, but execution time will be substantially longer.

## Reproducibility notes

Preprocessing and split-generation logic are embedded within the relevant scripts to make each experiment self-contained. Separate preprocessing or split-generation scripts are therefore not required to reproduce the reported results.

All scripts use fixed random seeds where applicable. The SMS experiments use a shared stratified development/holdout split and 5-fold cross-validation protocol. The URL experiments use a shared rare-class stratified development/holdout split and 5-fold cross-validation protocol.

The generated outputs include metrics, predictions, confusion matrices, ROC curves, and summary files depending on the script.

## Expected input files

The scripts expect the following datasets:

```text
data/ARA_SMS_Dataset_Final.csv
data/Smishing_Dataset_Final.csv
```

The Arabic SMS dataset should contain at least:

```text
Message
Label
```

The URL dataset should contain at least:

```text
url
label
```

Labels are normalized inside the scripts.

## Output folders

Recommended output folders:

```text
outputs/
outputs/runs/
outputs/plots/
outputs/splits/
outputs/smishing_results/
```

The scripts will create required output folders automatically where implemented.

## Environment

The experiments were implemented in Python and use common machine-learning and deep-learning libraries, including:

```text
numpy
pandas
scikit-learn
matplotlib
tensorflow
torch
transformers
datasets
xgboost
tldextract
requests
```

A complete dependency list should be provided in the root-level `requirements.txt` file.
