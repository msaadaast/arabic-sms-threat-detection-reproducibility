Modular Arabic SMS Spam and Smishing Detection Framework

This repository provides the reproducibility package for the paper:

A Modular and Deployment-Aware Framework for Arabic SMS Spam and Smishing Detection Using Text and URL Analysis

The study evaluates a modular Arabic SMS threat-detection framework with two complementary branches:

SMS text classification for Arabic ham/spam detection.
URL-based smishing analysis for benign/malicious URL detection.

The framework is evaluated under a leakage-controlled protocol using fixed development/holdout splits and 5-fold cross-validation.

Repository status during peer review

This is a review-stage reproducibility repository. It contains the experimental scripts, fixed split files, dependency file, and documentation required to inspect and reproduce the experimental workflow.

The full anonymized Arabic SMS corpus and smishing-oriented URL benchmark are not included in this review repository. They will be released publicly upon acceptance through a permanent repository with a persistent identifier, subject to final privacy screening and any applicable redistribution constraints.

Raw mobile-device exports, subscriber metadata, participant-identifying information, and non-anonymized collection artifacts will not be released.

Repository structure
README.md
requirements.txt
data/
README.md
scripts/
README.md
01_check_sms_duplicates_leakage.py
02_train_sms_ml_models.py
03_train_sms_1d_cnn.py
04_train_sms_distilbert.py
05_train_sms_arabert.py
06_train_sms_marbert.py
07_run_url_heuristic_engine.py
08_train_url_svm.py
09_train_url_xgboost.py
10_train_url_char_cnn.py
splits/
README.md
splits_holdout10_cv5_seed42.npz
ext_splits_holdout10_cv5_seed42.npz
url_shared_splits_holdout10_cv5.npz

Note: the data/ folder in this review repository contains only documentation. The full anonymized datasets will be added to the final public release after acceptance.

Dataset files expected for reproduction

When the anonymized datasets are released, the scripts expect the following local files.

1. Arabic SMS text dataset

Expected local path:

data/ARA_SMS_Dataset_Final.csv

This dataset is used for binary Arabic SMS ham/spam classification.

Required columns:

Column	Description
Message	SMS message text
Label	Message label: Ham or Spam
2. Smishing-oriented URL dataset

Expected local path:

data/Smishing_Dataset_Final.csv

This dataset is used for URL-based phishing/smishing analysis.

Required columns:

Column	Description
url	URL string
label	URL label: 0 for legitimate/benign and 1 for malicious/phishing
Privacy and anonymization

The Arabic SMS dataset was anonymized before modeling and will undergo final privacy screening before public release. Personally identifiable information appearing in message content was removed, masked, or replaced, including names, phone numbers, email addresses, physical addresses, bank identifiers, transaction-related references, and other sensitive details.

No subscriber metadata, device identifiers, sender or receiver numbers, SIM information, precise geolocation data, or other directly identifying metadata are included in the modeling dataset.

SMS-derived URLs were handled using the same privacy-preserving procedure. Where URLs contained personal or transaction-specific information, such information was removed or masked before inclusion.

Fixed split files

The splits/ folder contains the fixed split files used to reproduce the reported experiments.

File	Purpose
splits_holdout10_cv5_seed42.npz	Fixed development/holdout and 5-fold CV split for the primary Arabic SMS dataset
ext_splits_holdout10_cv5_seed42.npz	Fixed split for the external Arabic short-text methodological replication experiment
url_shared_splits_holdout10_cv5.npz	Fixed rare-class stratified split for the smishing-oriented URL dataset

All split files were generated using a fixed random seed of 42. These files preserve the exact development, holdout, and cross-validation partitions used in the manuscript so that model families are evaluated on identical data partitions.

Experimental scripts

The scripts/ folder contains the scripts used for model training, evaluation, and leakage checking.

Script	Purpose
01_check_sms_duplicates_leakage.py	Near-duplicate and post-split leakage analysis for the SMS dataset
02_train_sms_ml_models.py	Classical ML models for Arabic SMS classification
03_train_sms_1d_cnn.py	1D-CNN model for Arabic SMS classification
04_train_sms_distilbert.py	Multilingual DistilBERT fine-tuning
05_train_sms_arabert.py	AraBERT fine-tuning
06_train_sms_marbert.py	MARBERT fine-tuning
07_run_url_heuristic_engine.py	Interpretable URL heuristic engine and threshold evaluation
08_train_url_svm.py	Character-level TF-IDF + Linear SVM URL classifier
09_train_url_xgboost.py	Engineered-feature XGBoost URL classifier
10_train_url_char_cnn.py	Character-level CNN URL classifier

Detailed script-level instructions are provided in scripts/README.md.

Important path configuration note

The scripts currently use absolute paths corresponding to the original experimental environment. Before running any script, update DATA_PATH, OUTPUT_DIR, and any related path variables at the top of each script to match your local repository structure.

Recommended local structure:

project_root/
data/
ARA_SMS_Dataset_Final.csv
Smishing_Dataset_Final.csv
scripts/
splits/
outputs/

Recommended path settings:

Script type	DATA_PATH	OUTPUT_DIR
SMS text-classification scripts	data/ARA_SMS_Dataset_Final.csv	outputs/
URL-analysis scripts	data/Smishing_Dataset_Final.csv	outputs/smishing_results/
Duplicate / leakage-check script	data/ARA_SMS_Dataset_Final.csv	outputs/
Recommended execution order
Run the SMS text-classification scripts:

python scripts/02_train_sms_ml_models.py

python scripts/03_train_sms_1d_cnn.py

python scripts/04_train_sms_distilbert.py

python scripts/05_train_sms_arabert.py

python scripts/06_train_sms_marbert.py

Run the SMS leakage-checking script:

python scripts/01_check_sms_duplicates_leakage.py

Run the URL-analysis scripts:

python scripts/07_run_url_heuristic_engine.py

python scripts/08_train_url_svm.py

python scripts/09_train_url_xgboost.py

python scripts/10_train_url_char_cnn.py

Run 07_run_url_heuristic_engine.py before scripts 08_train_url_svm.py, 09_train_url_xgboost.py, and 10_train_url_char_cnn.py, because the heuristic script creates or uses the shared URL split file used by the learned URL models.

The transformer scripts require GPU acceleration for practical runtime. They can run on CPU, but execution time will be substantially longer.

Reproducibility notes

Preprocessing and split-generation logic are embedded within the relevant scripts to make each experiment self-contained. Separate preprocessing or split-generation scripts are therefore not required to reproduce the reported workflow.

All scripts use fixed random seeds where applicable. The SMS experiments use a shared stratified development/holdout split and 5-fold cross-validation protocol. The URL experiments use a shared rare-class stratified development/holdout split and 5-fold cross-validation protocol.

Generated outputs include summary metrics, prediction files, confusion matrices, ROC curves, and model-specific result files depending on the script.

Environment

The experiments were implemented in Python 3.10.19.

Main libraries include:

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
openpyxl

A complete dependency list is provided in requirements.txt.

Output directory

Generated results should be written to an outputs/ folder created locally by the user.

Recommended output structure:

outputs/
runs/
plots/
splits/
smishing_results/

The scripts create required output folders automatically where implemented.

Data availability

The full anonymized Arabic SMS corpus and smishing-oriented URL benchmark are not included in this review repository. They will be released publicly upon acceptance through a permanent repository with a persistent identifier.

This review repository provides the experimental scripts, fixed split files, dependency file, and documentation needed to inspect the methodology and reproduce the workflow once the anonymized datasets are available.

Raw mobile-device exports, subscriber metadata, participant-identifying information, and non-anonymized collection artifacts are not publicly available due to privacy and ethical constraints. Where redistribution of externally sourced URL records is restricted by source terms, reconstruction instructions and source references will be provided instead of redistributing restricted raw records.

Citation

If you use this repository, please cite the associated paper once published.
