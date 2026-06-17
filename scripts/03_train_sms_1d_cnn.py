
import os
import re
import time
import json
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATA_PATH  = "/home/mohamed/SMS_Project/data/ARA_SMS_Dataset_Final.csv"
OUTPUT_DIR = "/home/mohamed/SMS_Project"

# Protocol 
RANDOM_STATE        = 42
HOLDOUT_TEST_SIZE   = 0.10
N_SPLITS_CV         = 5

# Training
EPOCHS      = 6
BATCH_SIZE  = 32
VAL_SPLIT_IN_FOLD = 0.10
EARLY_STOP_PATIENCE = 2

# CNN Hyperparameters
MAX_WORDS       = 20000
MAX_LEN         = 100
EMBEDDING_DIM   = 128
FILTERS         = 128
KERNEL_SIZE     = 5
DROPOUT_RATE    = 0.5

# Outputs
RUN_TAG   = "cnn_unified_holdout10_cv5"
RUN_DIR   = os.path.join(OUTPUT_DIR, "runs", RUN_TAG)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots", RUN_TAG)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

LABEL_NAMES = {0: "ham", 1: "spam"}

# ---------------- REPRODUCIBILITY ----------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    tf.random.set_seed(seed)

seed_everything(RANDOM_STATE)

# ---------------- PREPROCESSING ----------------
AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = AR_DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("[يى]", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(r"[^ء-ي0-9A-Za-z\u0660-\u0669\s@#\$%\^\&\*\(\)\-_=+\:;,.?!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def error_type(true_label: int, pred_label: int) -> str:
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    return "FN"

def save_predictions_csv(out_path: str, messages, y_true, y_pred, prob_spam):
    msgs = list(messages)
    y_true = list(map(int, list(y_true)))
    y_pred = list(map(int, list(y_pred)))
    prob_spam = list(map(float, list(prob_spam)))

    df_out = pd.DataFrame({
        "Message": msgs,
        "True_Label_Bin": y_true,
        "True_Label": [LABEL_NAMES.get(v, str(v)) for v in y_true],
        "Pred_Label_Bin": y_pred,
        "Pred_Label": [LABEL_NAMES.get(v, str(v)) for v in y_pred],
        "Prob_Spam": prob_spam,
        "Error_Type": [error_type(t, p) for t, p in zip(y_true, y_pred)],
    })

    df_out["Abs_Conf_Delta"] = (df_out["Prob_Spam"] - 0.5).abs()
    df_out = df_out.sort_values(by=["Error_Type", "Abs_Conf_Delta"], ascending=[True, True]).drop(columns=["Abs_Conf_Delta"])

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    mis_path = out_path.replace(".csv", "_misclassified.csv")
    df_out[df_out["True_Label_Bin"] != df_out["Pred_Label_Bin"]].to_csv(mis_path, index=False, encoding="utf-8-sig")

def format_mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }

# ---------------- CNN MODEL ----------------
def build_cnn_model():
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def fit_tokenizer_on_texts(texts: np.ndarray) -> Tokenizer:
    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(list(texts))
    return tok

def vectorize_texts(tok: Tokenizer, texts: np.ndarray) -> np.ndarray:
    seq = tok.texts_to_sequences(list(texts))
    return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

def plot_training_curves(history, out_path: str, title_prefix: str):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Val Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("accuracy", []), label="Train Acc")
    plt.plot(history.history.get("val_accuracy", []), label="Val Acc")
    plt.title(f"{title_prefix} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

@dataclass
class FoldResult:
    fold: int
    train_size: int
    val_size: int
    train_time_sec: float
    eval_time_sec: float
    metrics: Dict[str, float]

# ---------------- DATA LOADING ----------------
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Message", "Label"]).reset_index(drop=True)
df["Label"] = df["Label"].astype(str).str.strip().str.lower().map(lambda x: "spam" if "spam" in x else "ham")
df["clean"] = df["Message"].apply(normalize_arabic)
df["label_bin"] = (df["Label"] == "spam").astype(int)

print("Data distribution:", df["Label"].value_counts().to_dict())
print(f"Total samples: {len(df)}")

X_all = df["clean"].values
y_all = df["label_bin"].values

# ---------------- SHARED SPLITS (HOLDOUT + CV FOLDS) ----------------
SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)
SPLITS_PATH = os.path.join(
    SPLITS_DIR,
    f"splits_holdout{int(HOLDOUT_TEST_SIZE*100)}_cv{N_SPLITS_CV}_seed{RANDOM_STATE}.npz"
)

def load_or_create_splits(labels: np.ndarray):
    n = len(labels)

    if os.path.exists(SPLITS_PATH):
        data = np.load(SPLITS_PATH, allow_pickle=True)
        holdout_idx = data["holdout_idx"]
        dev_idx = data["dev_idx"]
        folds = data["folds"]

        if len(holdout_idx) + len(dev_idx) != n:
            raise ValueError("Split file does not match current dataset length/order.")
        if folds.shape[0] != N_SPLITS_CV:
            raise ValueError("Split file does not match N_SPLITS_CV.")

        return dev_idx, holdout_idx, folds

    sss = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_TEST_SIZE, random_state=RANDOM_STATE)
    dev_idx, holdout_idx = next(sss.split(np.zeros(n), labels))

    skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for tr_rel, va_rel in skf.split(np.zeros(len(dev_idx)), labels[dev_idx]):
        tr_idx = dev_idx[tr_rel]
        va_idx = dev_idx[va_rel]
        folds.append((tr_idx, va_idx))
    folds = np.array(folds, dtype=object)

    np.savez_compressed(SPLITS_PATH, dev_idx=dev_idx, holdout_idx=holdout_idx, folds=folds)
    print("✓ Created shared split file:", SPLITS_PATH)
    return dev_idx, holdout_idx, folds

dev_idx, holdout_idx, folds = load_or_create_splits(y_all)

X_dev, y_dev = X_all[dev_idx], y_all[dev_idx]
X_test, y_test = X_all[holdout_idx], y_all[holdout_idx]

print(f"\nHoldout test size: {len(X_test)} ({HOLDOUT_TEST_SIZE*100:.0f}%)")
print(f"Development size:  {len(X_dev)} ({(1-HOLDOUT_TEST_SIZE)*100:.0f}%)")

# ---------------- STRATIFIED K-FOLD CV (shared folds) ----------------
fold_results: List[FoldResult] = []

print(f"\nRunning Stratified {N_SPLITS_CV}-Fold CV on development set...\n")

for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
    print(f"--- Fold {fold_idx}/{N_SPLITS_CV} ---")
    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)

    seed_everything(RANDOM_STATE + fold_idx)

   
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_va, y_va = X_all[val_idx], y_all[val_idx]

    # Fit tokenizer ONLY on training fold
    tok = fit_tokenizer_on_texts(X_tr)
    X_tr_pad = vectorize_texts(tok, X_tr)
    X_va_pad = vectorize_texts(tok, X_va)

    model = build_cnn_model()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=0
    )

    fold_out_dir = os.path.join(RUN_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_out_dir, exist_ok=True)

    t0 = time.time()
    history = model.fit(
        X_tr_pad, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT_IN_FOLD,
        callbacks=[early_stop],
        verbose=1
    )
    train_time = time.time() - t0

    plot_training_curves(
        history,
        os.path.join(PLOTS_DIR, f"training_curves_fold_{fold_idx}.png"),
        title_prefix=f"1D-CNN Fold {fold_idx}"
    )

    t1 = time.time()
    y_va_prob = model.predict(X_va_pad, verbose=0).flatten()
    y_va_pred = (y_va_prob > 0.5).astype(int)
    eval_time = time.time() - t1

    fold_metrics = compute_metrics(y_va, y_va_pred, y_va_prob)

    # Save fold predictions + confusion counts (aligned naming)
    fold_pred_path = os.path.join(fold_out_dir, "fold_val_predictions.csv")
    save_predictions_csv(fold_pred_path, X_va, y_va, y_va_pred, y_va_prob)

    cm_fold = confusion_matrix(y_va, y_va_pred)
    if cm_fold.shape == (2, 2):
        tn, fp, fn, tp = cm_fold.ravel()
    else:
        tn = fp = fn = tp = 0

    with open(os.path.join(fold_out_dir, "fold_val_confusion_counts.json"), "w", encoding="utf-8") as f:
        json.dump({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}, f, ensure_ascii=False, indent=2)

    fold_results.append(FoldResult(
        fold=fold_idx,
        train_size=len(X_tr),
        val_size=len(X_va),
        train_time_sec=train_time,
        eval_time_sec=eval_time,
        metrics=fold_metrics
    ))

    print(
        f"Fold {fold_idx} metrics: "
        f"Acc={fold_metrics['accuracy']:.4f}, "
        f"P={fold_metrics['precision']:.4f}, "
        f"R={fold_metrics['recall']:.4f}, "
        f"F1={fold_metrics['f1']:.4f}, "
        f"AUC={fold_metrics['roc_auc']:.4f} | "
        f"Train={format_mmss(train_time)}, Eval={format_mmss(eval_time)}"
    )
    print()

# ---------------- AGGREGATE CV RESULTS ----------------
def mean_std(values: List[float]) -> Tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
agg = {}
for k in metrics_keys:
    vals = [fr.metrics[k] for fr in fold_results]
    m, s = mean_std(vals)
    agg[k] = {"mean": m, "std": s, "per_fold": vals}

train_times = [fr.train_time_sec for fr in fold_results]
eval_times  = [fr.eval_time_sec  for fr in fold_results]
agg["train_time_sec"] = {"mean": float(np.mean(train_times)), "std": float(np.std(train_times, ddof=1)) if len(train_times)>1 else 0.0}
agg["eval_time_sec"]  = {"mean": float(np.mean(eval_times)),  "std": float(np.std(eval_times,  ddof=1)) if len(eval_times)>1 else 0.0}

cv_summary_path = os.path.join(RUN_DIR, "cv_summary.json")
with open(cv_summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": "1D-CNN",
        "run_tag": RUN_TAG,
        "config": {
            "random_state": RANDOM_STATE,
            "holdout_test_size": HOLDOUT_TEST_SIZE,
            "n_splits_cv": N_SPLITS_CV,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_words": MAX_WORDS,
            "max_len": MAX_LEN,
            "embedding_dim": EMBEDDING_DIM,
            "filters": FILTERS,
            "kernel_size": KERNEL_SIZE,
            "dropout_rate": DROPOUT_RATE,
            "val_split_in_fold": VAL_SPLIT_IN_FOLD,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        "cv": {
            "aggregate": agg,
            "folds": [asdict(fr) for fr in fold_results],
        },
    }, f, ensure_ascii=False, indent=2)

print("============================================================")
print(f"CV RESULTS (Dev set, Stratified {N_SPLITS_CV}-Fold) — 1D-CNN")
print("============================================================")
for k in metrics_keys:
    print(f"{k.upper():<10}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
print(f"{'TRAIN':<10}: {format_mmss(agg['train_time_sec']['mean'])} (mean)")
print(f"{'EVAL':<10}: {format_mmss(agg['eval_time_sec']['mean'])} (mean)")
print(f"\nSaved CV summary to: {cv_summary_path}\n")

# ---------------- FINAL TRAIN ON FULL DEV, EVAL ON HOLDOUT TEST ----------------
print("Training final CNN on full development set, then evaluating on untouched holdout test...")

seed_everything(RANDOM_STATE + 999)

tok_final = fit_tokenizer_on_texts(X_dev)
X_dev_pad  = vectorize_texts(tok_final, X_dev)
X_test_pad = vectorize_texts(tok_final, X_test)

final_model = build_cnn_model()
early_stop_final = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=True,
    verbose=0
)

t0 = time.time()
final_history = final_model.fit(
    X_dev_pad, y_dev,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT_IN_FOLD,
    callbacks=[early_stop_final],
    verbose=1
)
final_train_time = time.time() - t0

plot_training_curves(
    final_history,
    os.path.join(PLOTS_DIR, "training_curves_final.png"),
    title_prefix="1D-CNN Final (Dev)"
)

t1 = time.time()
y_test_prob = final_model.predict(X_test_pad, verbose=0).flatten()
y_test_pred = (y_test_prob > 0.5).astype(int)
final_eval_time = time.time() - t1

test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

print("============================================================")
print("HOLDOUT TEST RESULTS (Untouched test set)")
print("============================================================")
print(
    f"ACC={test_metrics['accuracy']:.4f} | "
    f"P={test_metrics['precision']:.4f} | "
    f"R={test_metrics['recall']:.4f} | "
    f"F1={test_metrics['f1']:.4f} | "
    f"AUC={test_metrics['roc_auc']:.4f}"
)
print(f"Train time: {format_mmss(final_train_time)} | Eval time: {format_mmss(final_eval_time)}\n")

# Save holdout metrics (aligned)
holdout_metrics_path = os.path.join(RUN_DIR, "holdout_test_metrics.json")
with open(holdout_metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": "1D-CNN",
        "run_tag": RUN_TAG,
        "holdout_test": test_metrics,
        "final_train_time_sec": float(final_train_time),
        "final_eval_time_sec": float(final_eval_time),
    }, f, ensure_ascii=False, indent=2)

# Save holdout predictions (aligned)
holdout_pred_csv = os.path.join(RUN_DIR, "holdout_predictions.csv")
save_predictions_csv(holdout_pred_csv, X_test, y_test, y_test_pred, y_test_prob)

# Save holdout confusion counts (aligned)
cm_counts = confusion_matrix(y_test, y_test_pred)
if cm_counts.shape == (2, 2):
    tn, fp, fn, tp = cm_counts.ravel()
else:
    tn = fp = fn = tp = 0

holdout_conf_path = os.path.join(RUN_DIR, "holdout_confusion_counts.json")
with open(holdout_conf_path, "w", encoding="utf-8") as f:
    json.dump({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}, f, ensure_ascii=False, indent=2)

# ---------------- PLOTS (holdout) ----------------
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm.astype(int), annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title(f"Confusion Matrix — {RUN_TAG}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cm_holdout.png"), dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title(f"ROC Curve — {RUN_TAG}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_holdout.png"), dpi=150)
plt.close()

print("✓ Holdout predictions saved to:", holdout_pred_csv)
print("✓ Holdout confusion counts saved to:", holdout_conf_path)
print("✓ Holdout metrics saved to:", holdout_metrics_path)
print("✓ Plots saved to:", PLOTS_DIR)
print("✓ Done.")