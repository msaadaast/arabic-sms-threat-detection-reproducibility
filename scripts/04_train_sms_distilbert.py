
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
import torch
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

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATA_PATH   = "/home/mohamed/SMS_Project/data/ARA_SMS_Dataset_Final.csv"
OUTPUT_DIR  = "/home/mohamed/SMS_Project"

MODEL_NAME  = "distilbert-base-multilingual-cased"

# Protocol 
RANDOM_STATE        = 42
HOLDOUT_TEST_SIZE   = 0.10
N_SPLITS_CV         = 5

# Training
EPOCHS              = 3
LR                  = 2e-5
MAX_LEN             = 128
BATCH_SIZE          = 8
GRAD_ACCUM_STEPS    = 2
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.06
EARLY_STOP_PATIENCE = 2
EVAL_STRATEGY       = "epoch"

# Outputs
RUN_TAG     = "distilbert_unified_holdout10_cv5"
RUN_DIR     = os.path.join(OUTPUT_DIR, "runs", RUN_TAG)
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots", RUN_TAG)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = bool(torch.cuda.is_available())

LABEL_NAMES = {0: "ham", 1: "spam"}

# ---------------- REPRODUCIBILITY ----------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

seed_everything(RANDOM_STATE)

# ---------------- PREPROCESSING ----------------
# Arabic diacritics range: U+064B..U+0652 plus Tatweel U+0640
AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove diacritics / tatweel
    text = AR_DIACRITICS.sub("", text)
    # Normalize
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("[يى]", "ي", text)
    text = re.sub("ة", "ه", text)
    # Keep Arabic letters, digits, latin, punctuation
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

    # Put errors first, then near decision boundary
    df_out["Abs_Conf_Delta"] = (df_out["Prob_Spam"] - 0.5).abs()
    df_out = df_out.sort_values(by=["Error_Type", "Abs_Conf_Delta"], ascending=[True, True]).drop(columns=["Abs_Conf_Delta"])

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    mis_path = out_path.replace(".csv", "_misclassified.csv")
    df_out[df_out["True_Label_Bin"] != df_out["Pred_Label_Bin"]].to_csv(mis_path, index=False, encoding="utf-8-sig")

def format_mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

# ---------------- METRICS ----------------
def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_metrics_from_logits(logits, labels)

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

    # Create holdout split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_TEST_SIZE, random_state=RANDOM_STATE)
    dev_idx, holdout_idx = next(sss.split(np.zeros(n), labels))

    # Create CV folds on dev set
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

dev_idx, holdout_idx, folds = load_or_create_splits(y_all)

X_dev, y_dev = X_all[dev_idx], y_all[dev_idx]
X_test, y_test = X_all[holdout_idx], y_all[holdout_idx]

print(f"\nHoldout test size: {len(X_test)} ({HOLDOUT_TEST_SIZE*100:.0f}%)")
print(f"Development size:  {len(X_dev)} ({(1-HOLDOUT_TEST_SIZE)*100:.0f}%)")

# ---------------- TOKENIZER ----------------
print("\nLoading tokenizer:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def make_hf_dataset(texts: np.ndarray, labels: np.ndarray) -> Dataset:
    enc = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN)
    return Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": list(labels),
    })

@dataclass
class FoldResult:
    fold: int
    train_size: int
    val_size: int
    train_time_sec: float
    eval_time_sec: float
    metrics: Dict[str, float]

# ---------------- STRATIFIED K-FOLD CV (shared folds) ----------------
fold_results: List[FoldResult] = []

print(f"\nRunning Stratified {N_SPLITS_CV}-Fold CV on development set...\n")

for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
    print(f"--- Fold {fold_idx}/{N_SPLITS_CV} ---")
    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_va, y_va = X_all[val_idx], y_all[val_idx]

    train_ds = make_hf_dataset(X_tr, y_tr)
    val_ds   = make_hf_dataset(X_va, y_va)

    # Fresh model each fold
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    fold_out_dir = os.path.join(RUN_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=fold_out_dir,
        evaluation_strategy=EVAL_STRATEGY,
        save_strategy=EVAL_STRATEGY,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=USE_FP16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=os.path.join(fold_out_dir, "logs"),
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=True,
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    # Train
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Predict (val)
    t1 = time.time()
    pred = trainer.predict(val_ds)
    eval_time = time.time() - t1

    val_logits = pred.predictions
    val_true   = pred.label_ids
    val_pred   = np.argmax(val_logits, axis=1)
    val_prob_spam = torch.nn.functional.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()

    fold_metrics = compute_metrics_from_logits(val_logits, val_true)

    # Save fold predictions + confusion counts
    fold_pred_path = os.path.join(fold_out_dir, "fold_val_predictions.csv")
    save_predictions_csv(fold_pred_path, X_va, val_true, val_pred, val_prob_spam)

    cm_fold = confusion_matrix(val_true, val_pred)
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
        "model": MODEL_NAME,
        "run_tag": RUN_TAG,
        "config": {
            "random_state": RANDOM_STATE,
            "holdout_test_size": HOLDOUT_TEST_SIZE,
            "n_splits_cv": N_SPLITS_CV,
            "epochs": EPOCHS,
            "lr": LR,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "fp16": USE_FP16,
        },
        "cv": {
            "aggregate": agg,
            "folds": [asdict(fr) for fr in fold_results],
        },
    }, f, ensure_ascii=False, indent=2)

print("============================================================")
print(f"CV RESULTS (Dev set, Stratified {N_SPLITS_CV}-Fold) — {MODEL_NAME}")
print("============================================================")
for k in metrics_keys:
    print(f"{k.upper():<10}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
print(f"{'TRAIN':<10}: {format_mmss(agg['train_time_sec']['mean'])} (mean)")
print(f"{'EVAL':<10}: {format_mmss(agg['eval_time_sec']['mean'])} (mean)")
print(f"\nSaved CV summary to: {cv_summary_path}\n")

# ---------------- FINAL TRAIN ON FULL DEV, EVAL ON HOLDOUT TEST ----------------
print("Training final model on full development set, then evaluating on untouched holdout test...")

final_train_ds = make_hf_dataset(X_dev, y_dev)
final_test_ds  = make_hf_dataset(X_test, y_test)

final_out_dir = os.path.join(RUN_DIR, "final_model")
os.makedirs(final_out_dir, exist_ok=True)

final_args = TrainingArguments(
    output_dir=final_out_dir,
    evaluation_strategy=EVAL_STRATEGY,
    save_strategy=EVAL_STRATEGY,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    fp16=USE_FP16,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    logging_dir=os.path.join(final_out_dir, "logs"),
    logging_steps=50,
    report_to="none",
    dataloader_pin_memory=True,
    seed=RANDOM_STATE,
)

final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

final_trainer = Trainer(
    model=final_model,
    args=final_args,
    train_dataset=final_train_ds,
    eval_dataset=final_test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
)

t0 = time.time()
final_trainer.train()
final_train_time = time.time() - t0

t1 = time.time()
test_pred = final_trainer.predict(final_test_ds)
final_eval_time = time.time() - t1

logits = test_pred.predictions
labels = test_pred.label_ids
pred_labels = np.argmax(logits, axis=1)
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

test_metrics = compute_metrics_from_logits(logits, labels)

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

# Save holdout metrics
holdout_metrics_path = os.path.join(RUN_DIR, "holdout_test_metrics.json")
with open(holdout_metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME,
        "run_tag": RUN_TAG,
        "holdout_test": test_metrics,
        "final_train_time_sec": float(final_train_time),
        "final_eval_time_sec": float(final_eval_time),
    }, f, ensure_ascii=False, indent=2)

# Save holdout predictions for later misclassification analysis
holdout_pred_csv = os.path.join(RUN_DIR, "holdout_predictions.csv")
save_predictions_csv(holdout_pred_csv, X_test, labels, pred_labels, probs)

# Confusion matrix components on holdout
cm_counts = confusion_matrix(labels, pred_labels)
if cm_counts.shape == (2, 2):
    tn, fp, fn, tp = cm_counts.ravel()
else:
    tn = fp = fn = tp = 0

holdout_conf_path = os.path.join(RUN_DIR, "holdout_confusion_counts.json")
with open(holdout_conf_path, "w", encoding="utf-8") as f:
    json.dump({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}, f, ensure_ascii=False, indent=2)

# ---------------- PLOTS (holdout) ----------------
cm = confusion_matrix(labels, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm.astype(int), annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title(f"Confusion Matrix — {RUN_TAG}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cm_holdout.png"), dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(labels, probs)
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