import os
import re
import time
import json
import math
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "/home/mohamed/SMS_Project/data/Smishing_Dataset_Final.csv"
OUTPUT_DIR = "/home/mohamed/SMS_Project/smishing_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_XGBoost_detailed.xlsx")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_XGBoost_summary.csv")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_XGBoost_summary.json")
FEATURE_IMPORTANCE_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_XGBoost_feature_importance.csv")

TEXT_COL = "url"
LABEL_COL = "label"

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.10
CV_FOLDS = 5

SPLIT_FILE = os.path.join(OUTPUT_DIR, "url_shared_splits_holdout10_cv5.npz")

# ============================================================
# Setup folders
# ============================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("URL XGBoost Script starting...")
print(f"DATA_PATH  : {DATA_PATH}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"PLOTS_DIR  : {PLOTS_DIR}")
print("=" * 60)

# ============================================================
# Constants / dictionaries
# ============================================================
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "account", "secure", "bank",
    "reset", "confirm", "otp", "password", "reward", "winner",
    "free", "urgent", "warning", "gift", "money", "admin",
    "redeem", "claim", "payment", "wallet", "unlock"
]

KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rebrand.ly", "bit.do", "onelink.to",
    "rb.gy", "shorturl.at", "wa.link", "t.ly", "2u.pw",
    "tiny.cc", "bitly.ws", "shr.mx"
}

COMMON_BENIGN_TLDS = {
    "com", "org", "net", "edu", "gov", "co", "io", "app", "info",
    "biz", "me", "tv", "cc", "eg", "com.eg", "org.eg", "edu.eg",
    "gov.eg", "co.uk", "org.uk", "edu.au", "org.au", "net.au",
    "ae", "sa", "qa", "bh", "kw", "om", "jo", "ma", "tn"
}

SPECIAL_CHARS_PATTERN = r"[@%;{}\[\]\|\\=_&\?\-\.\/:]"

# ============================================================
# Label normalization
# ============================================================
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

# ============================================================
# URL helpers
# ============================================================
def safe_parse_url(url: str):
    """
    Robust URL parsing.
    Adds scheme if missing to ensure urlparse works consistently.
    """
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        url = "http://" + url
    return urlparse(url)

def extract_domain_parts(netloc: str):
    """
    Returns:
    - full domain
    - host without port
    - list of labels
    """
    host = netloc.lower().strip()
    if ":" in host:
        host = host.split(":")[0]
    labels = [p for p in host.split(".") if p]
    return host, labels

def approximate_tld(labels):
    """
    Approximate TLD extraction.
    Uses last label, or last two labels for common multi-part forms.
    """
    if not labels:
        return ""
    if len(labels) >= 2:
        last_two = ".".join(labels[-2:])
        if last_two in COMMON_BENIGN_TLDS:
            return last_two
    return labels[-1]

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def is_ip_address(host: str) -> int:
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        return 1
    return 0

def count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))

def contains_keyword(text: str, keywords) -> int:
    text = text.lower()
    return int(any(k in text for k in keywords))


HEURISTIC_DETAIL_PATH = "/home/mohamed/SMS_Project/smishing_results/Smishing_heuristic_detailed.xlsx"

dynamic_feature_map = {}

def load_dynamic_features():
    global dynamic_feature_map
    if HEURISTIC_DETAIL_PATH is None:
        print("No heuristic detail path provided. Optional dynamic features will default to 0.")
        return

    path = HEURISTIC_DETAIL_PATH
    if not os.path.exists(path):
        print(f"Heuristic detail file not found: {path}")
        print("Optional dynamic features will default to 0.")
        return

    print(f"Loading optional dynamic features from: {path}")

    if path.endswith(".xlsx"):
        dyn_df = pd.read_excel(path)
    else:
        dyn_df = pd.read_csv(path)

    if "url" not in dyn_df.columns:
        print("No 'url' column found in heuristic detail file.")
        print("Optional dynamic features will default to 0.")
        return

    dyn_df["url"] = dyn_df["url"].astype(str).str.strip()

    possible_dns = [c for c in dyn_df.columns if "dns" in c.lower()]
    possible_ssl = [c for c in dyn_df.columns if "ssl" in c.lower() or "tls" in c.lower()]
    possible_http = [c for c in dyn_df.columns if "response" in c.lower() or "http" in c.lower() or "head" in c.lower()]

    dns_col = possible_dns[0] if possible_dns else None
    ssl_col = possible_ssl[0] if possible_ssl else None
    http_col = possible_http[0] if possible_http else None

    for _, row in dyn_df.iterrows():
        dynamic_feature_map[row["url"]] = {
            "dns_resolution_failure": int(row[dns_col]) if dns_col and pd.notna(row[dns_col]) else 0,
            "invalid_ssl": int(row[ssl_col]) if ssl_col and pd.notna(row[ssl_col]) else 0,
            "http_response_failure": int(row[http_col]) if http_col and pd.notna(row[http_col]) else 0,
        }

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

# ============================================================
# Feature engineering
# ============================================================
def engineer_features(url: str):
    raw_url = str(url).strip()
    parsed = safe_parse_url(raw_url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""
    full_tail = (path + ("?" + query if query else ""))

    host, labels = extract_domain_parts(netloc)
    tld = approximate_tld(labels)
    dom = host
    dom_len = len(dom)
    tld_len = len(tld)


    subdom_cnt = max(len(labels) - 2, 0) if len(labels) >= 2 else 0

    url_len = len(raw_url)
    letter_cnt = sum(c.isalpha() for c in raw_url)
    digit_cnt = sum(c.isdigit() for c in raw_url)
    special_cnt = count_matches(SPECIAL_CHARS_PATTERN, raw_url)

    eq_cnt = raw_url.count("=")
    qm_cnt = raw_url.count("?")
    amp_cnt = raw_url.count("&")
    dot_cnt = raw_url.count(".")
    dash_cnt = raw_url.count("-")
    under_cnt = raw_url.count("_")
    slash_cnt = raw_url.count("/")

    entropy = shannon_entropy(raw_url)

    path_len = len(path)
    query_len = len(query)

    letter_ratio = letter_cnt / url_len if url_len > 0 else 0.0
    digit_ratio = digit_cnt / url_len if url_len > 0 else 0.0
    spec_ratio = special_cnt / url_len if url_len > 0 else 0.0

    is_https = int(scheme == "https")
    has_http_only = int(scheme == "http")
    is_ip = is_ip_address(host)

    has_shortener = int(
        host in KNOWN_SHORTENERS or any(short in host for short in KNOWN_SHORTENERS)
    )

    has_suspicious_keyword = contains_keyword(raw_url, SUSPICIOUS_KEYWORDS)
    has_path_keyword = contains_keyword(full_tail, SUSPICIOUS_KEYWORDS)

    has_uncommon_tld = int(tld not in COMMON_BENIGN_TLDS)

    # Optional dynamic features
    dyn = dynamic_feature_map.get(raw_url, {})
    dns_resolution_failure = int(dyn.get("dns_resolution_failure", 0))
    invalid_ssl = int(dyn.get("invalid_ssl", 0))
    http_response_failure = int(dyn.get("http_response_failure", 0))

    return {
        "url_len": url_len,
        "dom_len": dom_len,
        "tld_len": tld_len,
        "subdom_cnt": subdom_cnt,
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": eq_cnt,
        "qm_cnt": qm_cnt,
        "amp_cnt": amp_cnt,
        "dot_cnt": dot_cnt,
        "dash_cnt": dash_cnt,
        "under_cnt": under_cnt,
        "slash_cnt": slash_cnt,
        "path_len": path_len,
        "query_len": query_len,
        "entropy": entropy,
        "letter_ratio": letter_ratio,
        "digit_ratio": digit_ratio,
        "spec_ratio": spec_ratio,
        "is_https": is_https,
        "has_http_only": has_http_only,
        "is_ip": is_ip,
        "has_shortener": has_shortener,
        "has_suspicious_keyword": has_suspicious_keyword,
        "has_path_keyword": has_path_keyword,
        "has_uncommon_tld": has_uncommon_tld,
        "dns_resolution_failure": dns_resolution_failure,
        "invalid_ssl": invalid_ssl,
        "http_response_failure": http_response_failure,
    }

# ============================================================
# Plot helpers
# ============================================================
def save_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Smishing URL XGBoost - Holdout Confusion Matrix")
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
    ax.set_title("Smishing URL XGBoost - Holdout ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_feature_importance_plot(feature_importance_df, out_path):
    top_df = feature_importance_df.head(20).copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"][::-1], top_df["importance"][::-1])
    ax.set_title("XGBoost Feature Importance (Top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Load dynamic features if provided
# ============================================================
load_dynamic_features()

# ============================================================
# Load dataset
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
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

df[LABEL_COL] = df[LABEL_COL].apply(normalize_label)

print(f"Total samples after cleaning: {len(df)}")
print("Label distribution:", dict(df[LABEL_COL].value_counts().sort_index()))

# ============================================================
# Engineer features
# ============================================================
print("\nEngineering URL features...")
feature_rows = []
for url in df[TEXT_COL]:
    feature_rows.append(engineer_features(url))

features_df = pd.DataFrame(feature_rows)
full_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

feature_cols = list(features_df.columns)

print(f"Engineered {len(feature_cols)} features:")
print(feature_cols)

# ============================================================
# Shared split usage
# ============================================================
X_all = full_df[feature_cols]
y_all = full_df[LABEL_COL].values

dev_idx, holdout_idx, folds = load_existing_shared_splits(y_all)

X_dev = X_all.iloc[dev_idx]
X_holdout = X_all.iloc[holdout_idx]
y_dev = y_all[dev_idx]
y_holdout = y_all[holdout_idx]
url_dev = full_df[TEXT_COL].values[dev_idx]
url_holdout = full_df[TEXT_COL].values[holdout_idx]

print(f"\nDevelopment set: {len(X_dev)}")
print(f"Holdout set    : {len(X_holdout)}")

# ============================================================
# Model
# ============================================================
model_name = "XGBoost_URL_Features"

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("clf", xgb_model),
])

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

print("\n" + "=" * 60)
print("Running CV + Holdout evaluation...")
print("=" * 60)

t0 = time.time()
cv_results = cross_validate(
    pipeline,
    X_all,
    y_all,
    cv=folds,
    scoring=cv_scoring,
    return_train_score=False,
    n_jobs=1,  # safer with xgboost internal threading
)
cv_total_time = time.time() - t0

# ============================================================
# Holdout training
# ============================================================
t1 = time.time()
pipeline.fit(X_dev, y_dev)
train_time = time.time() - t1

# ============================================================
# Holdout prediction
# ============================================================
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

print(f"\nHoldout Accuracy : {acc:.4f}")
print(f"Holdout Precision: {prec:.4f}")
print(f"Holdout Recall   : {rec:.4f}")
print(f"Holdout F1       : {f1:.4f}")
print(f"Holdout ROC-AUC  : {roc_auc:.4f}")
print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ============================================================
# Save plots
# ============================================================
cm_plot = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
roc_plot = os.path.join(PLOTS_DIR, f"{model_name}_roc_curve.png")
fi_plot = os.path.join(PLOTS_DIR, f"{model_name}_feature_importance.png")

save_confusion_matrix(confusion_matrix(y_holdout, y_pred), model_name, cm_plot)
save_roc_curve(y_holdout, y_score, model_name, roc_plot)

# ============================================================
# Feature importance
# ============================================================
clf = pipeline.named_steps["clf"]
importances = clf.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

feature_importance_df.to_csv(FEATURE_IMPORTANCE_OUTPUT, index=False)
save_feature_importance_plot(feature_importance_df, fi_plot)

# ============================================================
# Save outputs
# ============================================================
summary_df = pd.DataFrame([{
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
    "Num_Features": len(feature_cols),
}])

details_df = pd.DataFrame({
    "url": url_holdout,
    "true_label": y_holdout,
    "pred_label": y_pred,
    "pred_score_phishing": y_score,
})

engineered_dataset_df = pd.concat(
    [full_df[[TEXT_COL, LABEL_COL]].reset_index(drop=True),
     full_df[feature_cols].reset_index(drop=True)],
    axis=1
)

summary_df.to_csv(SUMMARY_OUTPUT, index=False)
summary_df.to_json(JSON_OUTPUT, orient="records", indent=2)

with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    details_df.to_excel(writer, sheet_name="Holdout_Details", index=False)
    feature_importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
    engineered_dataset_df.to_excel(writer, sheet_name="Engineered_Dataset", index=False)

print("\n" + "=" * 60)
print("FINAL XGBOOST URL SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

print("\nTop 20 Feature Importances:")
print(feature_importance_df.head(20).to_string(index=False))

print("\nSaved files:")
print(f"- Summary CSV        : {SUMMARY_OUTPUT}")
print(f"- Summary JSON       : {JSON_OUTPUT}")
print(f"- Excel file         : {EXCEL_OUTPUT}")
print(f"- Feature importance : {FEATURE_IMPORTANCE_OUTPUT}")
print(f"- Plots dir          : {PLOTS_DIR}")