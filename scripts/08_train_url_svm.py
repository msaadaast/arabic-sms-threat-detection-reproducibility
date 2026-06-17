import os
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "/home/mohamed/SMS_Project/data/Smishing_Dataset_Final.csv"
OUTPUT_DIR = "/home/mohamed/SMS_Project/smishing_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_ML_detailed.xlsx")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_ML_summary.csv")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_ML_summary.json")

TEXT_COL = "url"
LABEL_COL = "label"

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.10
CV_FOLDS = 5

SPLIT_FILE = os.path.join(OUTPUT_DIR, "url_shared_splits_holdout10_cv5.npz")

# Character n-gram TF-IDF settings
NGRAM_RANGE = (3, 5)
MIN_DF = 1
MAX_FEATURES = 50000
LOWERCASE = True

# ============================================================
# Setup folders
# ============================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("URL ML Unified Script starting...")
print(f"DATA_PATH  : {DATA_PATH}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"PLOTS_DIR  : {PLOTS_DIR}")
print("=" * 60)

# ============================================================
# Load and clean dataset
# ============================================================
df = pd.read_csv(DATA_PATH)

if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
    raise ValueError(
        f"Expected columns '{TEXT_COL}' and '{LABEL_COL}' not found. "
        f"Available columns: {list(df.columns)}"
    )

df = df[[TEXT_COL, LABEL_COL]].copy()
df = df.dropna(subset=[TEXT_COL, LABEL_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""]

# Drop duplicate URLs to avoid artificial inflation
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)


def normalize_label(x):
    s = str(x).strip().lower()
    if s in {"0", "benign", "legitimate", "ham", "safe"}:
        return 0
    if s in {"1", "phishing", "malicious", "spam", "smish", "smishing"}:
        return 1
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
    except Exception:
        pass
    raise ValueError(f"Unsupported label value: {x}")

df[LABEL_COL] = df[LABEL_COL].apply(normalize_label)

X_all = df[TEXT_COL].values
y_all = df[LABEL_COL].values

print(f"Total samples after cleaning: {len(df)}")
print("Label distribution:", dict(pd.Series(y_all).value_counts().sort_index()))

# ============================================================
# Shared split loader
# ============================================================
def load_existing_shared_splits(y_all, split_file=SPLIT_FILE):
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Shared split file not found: {split_file}\n"
            f"Run the heuristic script first to create it."
        )

    data = np.load(split_file, allow_pickle=True)

    saved_n = int(data["n_samples"])
    saved_y = data["y_all"]

    if saved_n != len(y_all) or not np.array_equal(saved_y, y_all):
        raise ValueError(
            "Shared split file does not match current dataset length/order/labels."
        )

    dev_idx = data["dev_idx"]
    holdout_idx = data["holdout_idx"]
    folds = data["folds"].tolist()

    return dev_idx, holdout_idx, folds

dev_idx, holdout_idx, folds = load_existing_shared_splits(y_all)

X_dev = X_all[dev_idx]
X_holdout = X_all[holdout_idx]
y_dev = y_all[dev_idx]
y_holdout = y_all[holdout_idx]

print(f"Development set: {len(X_dev)}")
print(f"Holdout set    : {len(X_holdout)}")

# ============================================================
# Models
# ============================================================
models = {
    "Linear_SVM": CalibratedClassifierCV(
        estimator=LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        method="sigmoid",
        cv=3,
    ),
}

def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=NGRAM_RANGE,
            lowercase=LOWERCASE,
            min_df=MIN_DF,
            max_features=MAX_FEATURES,
            sublinear_tf=True,
        )),
        ("clf", model),
    ])

# ============================================================
# Plot helpers
# ============================================================
def save_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Smishing URL SVM - Holdout Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    tick_marks = np.arange(2)
    classes = ["Benign (0)", "Phishing (1)"]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, y_score, model_name, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Smishing URL SVM - Holdout ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# CV evaluation
# ============================================================
cv_scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

summary_rows = []
detailed_holdout_rows = []

print("\n" + "=" * 60)
print("Running CV + Holdout evaluation...")
print("=" * 60)

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    pipeline = build_pipeline(model)

    # --------------------------
    # Cross-validation on shared folds
    # --------------------------
    t0 = time.time()
    cv_results = cross_validate(
        pipeline,
        X_all,
        y_all,
        cv=folds,
        scoring=cv_scoring,
        return_train_score=False,
        n_jobs=-1,
    )
    cv_total_time = time.time() - t0

    # --------------------------
    # Holdout training
    # --------------------------
    t1 = time.time()
    pipeline.fit(X_dev, y_dev)
    train_time = time.time() - t1

    # --------------------------
    # Holdout prediction
    # --------------------------
    t2 = time.time()
    y_pred = pipeline.predict(X_holdout)
    y_score = pipeline.predict_proba(X_holdout)[:, 1]
    eval_time = time.time() - t2

    acc = accuracy_score(y_holdout, y_pred)
    prec = precision_score(y_holdout, y_pred, zero_division=0)
    rec = recall_score(y_holdout, y_pred, zero_division=0)
    f1 = f1_score(y_holdout, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_holdout, y_score)

    tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred).ravel()

    print(f"Holdout Accuracy : {acc:.4f}")
    print(f"Holdout Precision: {prec:.4f}")
    print(f"Holdout Recall   : {rec:.4f}")
    print(f"Holdout F1       : {f1:.4f}")
    print(f"Holdout ROC-AUC  : {roc_auc:.4f}")
    print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Save plots
    cm_plot = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
    roc_plot = os.path.join(PLOTS_DIR, f"{model_name}_roc_curve.png")
    save_confusion_matrix(confusion_matrix(y_holdout, y_pred), model_name, cm_plot)
    save_roc_curve(y_holdout, y_score, model_name, roc_plot)

    # Summary
    summary_rows.append({
        "Model": model_name,
        "CV_Accuracy_Mean": np.mean(cv_results["test_accuracy"]),
        "CV_Accuracy_Std": np.std(cv_results["test_accuracy"]),
        "CV_Precision_Mean": np.mean(cv_results["test_precision"]),
        "CV_Precision_Std": np.std(cv_results["test_precision"]),
        "CV_Recall_Mean": np.mean(cv_results["test_recall"]),
        "CV_Recall_Std": np.std(cv_results["test_recall"]),
        "CV_F1_Mean": np.mean(cv_results["test_f1"]),
        "CV_F1_Std": np.std(cv_results["test_f1"]),
        "CV_ROCAUC_Mean": np.mean(cv_results["test_roc_auc"]),
        "CV_ROCAUC_Std": np.std(cv_results["test_roc_auc"]),
        "CV_Total_Time_s": cv_total_time,
        "Holdout_Accuracy": acc,
        "Holdout_Precision": prec,
        "Holdout_Recall": rec,
        "Holdout_F1": f1,
        "Holdout_ROCAUC": roc_auc,
        "Holdout_TN": tn,
        "Holdout_FP": fp,
        "Holdout_FN": fn,
        "Holdout_TP": tp,
        "Holdout_Train_Time_s": train_time,
        "Holdout_Eval_Time_s": eval_time,
        "N_Dev": len(X_dev),
        "N_Holdout": len(X_holdout),
        "Char_Ngram_Range": str(NGRAM_RANGE),
        "Max_Features": MAX_FEATURES,
    })

    # Detailed holdout predictions
    for url, true_label, pred_label, score in zip(X_holdout, y_holdout, y_pred, y_score):
        detailed_holdout_rows.append({
            "Model": model_name,
            "url": url,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "pred_score_phishing": float(score),
        })

# ============================================================
# Save outputs
# ============================================================
summary_df = pd.DataFrame(summary_rows)
details_df = pd.DataFrame(detailed_holdout_rows)

summary_df = summary_df.sort_values(by="Holdout_F1", ascending=False).reset_index(drop=True)

summary_df.to_csv(SUMMARY_OUTPUT, index=False)
summary_df.to_json(JSON_OUTPUT, orient="records", indent=2)

with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    details_df.to_excel(writer, sheet_name="Holdout_Details", index=False)
    df.to_excel(writer, sheet_name="Cleaned_Dataset", index=False)

print("\n" + "=" * 60)
print("FINAL URL ML SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

print("\nSaved files:")
print(f"- Summary CSV : {SUMMARY_OUTPUT}")
print(f"- Summary JSON: {JSON_OUTPUT}")
print(f"- Excel file  : {EXCEL_OUTPUT}")
print(f"- Plots dir   : {PLOTS_DIR}")