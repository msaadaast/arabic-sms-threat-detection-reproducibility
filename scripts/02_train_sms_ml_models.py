
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATA_PATH   = "/home/mohamed/SMS_Project/data/ARA_SMS_Dataset_Final.csv"
OUTPUT_DIR  = "/home/mohamed/SMS_Project"

RANDOM_STATE        = 42
HOLDOUT_TEST_SIZE   = 0.10
N_SPLITS_CV         = 5

TFIDF_MAX_FEATURES  = 20000
TFIDF_NGRAM         = (1, 2)

RUN_TAG   = "ml_unified_holdout10_cv5"
RUN_DIR   = os.path.join(OUTPUT_DIR, "runs", RUN_TAG)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots", RUN_TAG)

# ---- VERBOSE OUTPUT  ----
VERBOSE = True
def vprint(*args, **kwargs):
    if VERBOSE:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

# Ensure dirs exist
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

LABEL_NAMES = {0: "ham", 1: "spam"}

# ---------------- REPRODUCIBILITY ----------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

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

def confusion_counts(y_true, y_pred) -> Dict[str, int]:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    cc = confusion_counts(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        **cc,
    }

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def get_scores(model, X_vec):
    # Return a spam "probability-like" score for ROC and CSVs:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_vec)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X_vec)
        s = np.asarray(s)
        if s.ndim > 1:
            s = s[:, 0]
        return sigmoid(s)
    return model.predict(X_vec).astype(float)

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
        return dev_idx, holdout_idx, folds, True

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
    return dev_idx, holdout_idx, folds, False

# ---------------- DATA LOADING ----------------
vprint("\n============================================================")
vprint("ML Unified Script starting...")
vprint(f"DATA_PATH  : {DATA_PATH}")
vprint(f"OUTPUT_DIR : {OUTPUT_DIR}")
vprint(f"RUN_DIR    : {RUN_DIR}")
vprint(f"PLOTS_DIR  : {PLOTS_DIR}")
vprint("============================================================\n")

vprint("Loading dataset CSV...", flush=True)
df = pd.read_csv(DATA_PATH)
vprint(f"Loaded raw rows: {len(df)}")

df = df.dropna(subset=["Message", "Label"]).reset_index(drop=True)
vprint(f"After dropna: {len(df)} rows")

vprint("Normalizing labels + cleaning text...", flush=True)
df["Label"] = df["Label"].astype(str).str.strip().str.lower().map(lambda x: "spam" if "spam" in x else "ham")
df["clean"] = df["Message"].apply(normalize_arabic)
df["label_bin"] = (df["Label"] == "spam").astype(int)

vprint("Data distribution:", df["Label"].value_counts().to_dict())
vprint(f"Total samples: {len(df)}\n")

X_all = df["clean"].values
y_all = df["label_bin"].values

vprint("Loading/creating shared splits...", flush=True)
dev_idx, holdout_idx, folds, loaded_existing = load_or_create_splits(y_all)
vprint(f"Shared splits path: {SPLITS_PATH}")
vprint("Split file status :", "LOADED (existing)" if loaded_existing else "CREATED (new)")
vprint(f"Dev size    : {len(dev_idx)}")
vprint(f"Holdout size: {len(holdout_idx)}")
vprint(f"Num folds   : {len(folds)}\n")

X_dev, y_dev = X_all[dev_idx], y_all[dev_idx]
X_test, y_test = X_all[holdout_idx], y_all[holdout_idx]

# ---------------- MODELS ----------------
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

models = {
    "SVM": LinearSVC(random_state=RANDOM_STATE, class_weight="balanced"),
    "Logistic_Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
    "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced"),
    "Decision_Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE, algorithm="SAMME"),
    "Naive_Bayes": MultinomialNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

@dataclass
class FoldResult:
    fold: int
    train_size: int
    val_size: int
    train_time_sec: float
    eval_time_sec: float
    metrics: Dict[str, float]

def mean_std(values: List[float]) -> Tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

summary_rows = []
summary_json = {}

vprint("============================================================")
vprint("Running unified ML evaluation (shared holdout + shared CV folds)")
vprint("============================================================\n")

for model_name, base_model in models.items():
    try:
        vprint(f"\n{'='*70}")
        vprint(f"MODEL: {model_name}")
        vprint(f"TF-IDF: max_features={TFIDF_MAX_FEATURES}, ngram_range={TFIDF_NGRAM}")
        vprint(f"CV: {N_SPLITS_CV}-Fold on DEV | Holdout: {int(HOLDOUT_TEST_SIZE*100)}% untouched")
        vprint(f"{'='*70}")

        model_dir = os.path.join(RUN_DIR, "per_model", model_name)
        os.makedirs(model_dir, exist_ok=True)
        vprint(f"Outputs -> {model_dir}")

        fold_results: List[FoldResult] = []
        all_cm = np.zeros((2, 2), dtype=float)
        all_fpr = []
        all_tpr = []
        all_auc = []

        train_times = []
        eval_times = []

        # -------- CV using shared folds --------
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            vprint(f"\n[{model_name}] Fold {fold_idx}/{N_SPLITS_CV} starting...", flush=True)

            train_idx = np.array(train_idx, dtype=int)
            val_idx = np.array(val_idx, dtype=int)

            X_tr, y_tr = X_all[train_idx], y_all[train_idx]
            X_va, y_va = X_all[val_idx], y_all[val_idx]

            vprint(f"[{model_name}] Fold {fold_idx} vectorizing (fit on train fold only)...", flush=True)
            vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM)
            X_tr_vec = vectorizer.fit_transform(X_tr)
            X_va_vec = vectorizer.transform(X_va)

            model = clone(base_model)

            vprint(f"[{model_name}] Fold {fold_idx} training...", flush=True)
            t0 = time.time()
            model.fit(X_tr_vec, y_tr)
            train_time = time.time() - t0

            vprint(f"[{model_name}] Fold {fold_idx} evaluating...", flush=True)
            t1 = time.time()
            y_va_pred = model.predict(X_va_vec)
            y_va_prob = get_scores(model, X_va_vec)
            eval_time = time.time() - t1

            fold_metrics = compute_metrics(y_va, y_va_pred, y_va_prob)
            fold_results.append(FoldResult(
                fold=fold_idx,
                train_size=len(X_tr),
                val_size=len(X_va),
                train_time_sec=train_time,
                eval_time_sec=eval_time,
                metrics=fold_metrics
            ))

            train_times.append(train_time)
            eval_times.append(eval_time)

            fold_out_dir = os.path.join(model_dir, f"fold_{fold_idx}")
            os.makedirs(fold_out_dir, exist_ok=True)

            vprint(f"[{model_name}] Fold {fold_idx} saving predictions + confusion counts...", flush=True)
            fold_pred_path = os.path.join(fold_out_dir, "fold_val_predictions.csv")
            save_predictions_csv(fold_pred_path, X_va, y_va, y_va_pred, y_va_prob)

            with open(os.path.join(fold_out_dir, "fold_val_confusion_counts.json"), "w", encoding="utf-8") as f:
                json.dump({k: fold_metrics[k] for k in ["TN","FP","FN","TP"]}, f, ensure_ascii=False, indent=2)

            cm = confusion_matrix(y_va, y_va_pred)
            all_cm += cm

            fpr, tpr, _ = roc_curve(y_va, y_va_prob)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_auc.append(auc(fpr, tpr))

            vprint(
                f"[{model_name}] Fold {fold_idx} DONE | "
                f"Acc={fold_metrics['accuracy']:.4f} P={fold_metrics['precision']:.4f} "
                f"R={fold_metrics['recall']:.4f} F1={fold_metrics['f1']:.4f} AUC={fold_metrics['roc_auc']:.4f} | "
                f"TN={fold_metrics['TN']} FP={fold_metrics['FP']} FN={fold_metrics['FN']} TP={fold_metrics['TP']} | "
                f"Train={format_mmss(train_time)} Eval={format_mmss(eval_time)}"
            )

        # CV aggregate
        vprint(f"\n[{model_name}] Aggregating CV results...", flush=True)
        agg = {}
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            vals = [fr.metrics[k] for fr in fold_results]
            m, s = mean_std(vals)
            agg[k] = {"mean": m, "std": s, "per_fold": vals}

        agg["train_time_sec"] = {"mean": float(np.mean(train_times)), "std": float(np.std(train_times, ddof=1)) if len(train_times)>1 else 0.0, "per_fold": train_times}
        agg["eval_time_sec"]  = {"mean": float(np.mean(eval_times)),  "std": float(np.std(eval_times,  ddof=1)) if len(eval_times)>1 else 0.0, "per_fold": eval_times}

        cv_summary_path = os.path.join(model_dir, "cv_summary.json")
        with open(cv_summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_name,
                "run_tag": RUN_TAG,
                "config": {
                    "random_state": RANDOM_STATE,
                    "holdout_test_size": HOLDOUT_TEST_SIZE,
                    "n_splits_cv": N_SPLITS_CV,
                    "tfidf_max_features": TFIDF_MAX_FEATURES,
                    "tfidf_ngram": TFIDF_NGRAM,
                },
                "cv": {
                    "aggregate": agg,
                    "folds": [asdict(fr) for fr in fold_results],
                },
            }, f, ensure_ascii=False, indent=2)

        vprint(f"[{model_name}] Saved CV summary: {cv_summary_path}", flush=True)

        # -------- Plots: combined metrics/time across folds --------
        vprint(f"[{model_name}] Plotting CV history + CM + ROC...", flush=True)
        folds_x = np.arange(1, N_SPLITS_CV + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(folds_x, agg["accuracy"]["per_fold"], marker="o", label="Accuracy")
        axes[0].plot(folds_x, agg["precision"]["per_fold"], marker="s", label="Precision")
        axes[0].plot(folds_x, agg["recall"]["per_fold"], marker="^", label="Recall")
        axes[0].plot(folds_x, agg["f1"]["per_fold"], marker="d", label="F1")
        axes[0].set_xticks(folds_x)
        axes[0].set_ylim(0, 1.01)
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("Score")
        axes[0].set_title(f"{model_name} - Metrics Across Folds")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(folds_x, train_times, marker="o", label="Train Time (s)")
        axes[1].plot(folds_x, eval_times, marker="s", label="Eval Time (s)")
        axes[1].set_xticks(folds_x)
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("Seconds")
        axes[1].set_title(f"{model_name} - Train & Eval Time Across Folds")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        history_path = os.path.join(PLOTS_DIR, f"history_{model_name}_cv.png")
        plt.savefig(history_path, dpi=150)
        plt.close()

        cm_cv_path = os.path.join(PLOTS_DIR, f"cm_cv_{model_name}.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(all_cm.astype(int), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        plt.title(f"{model_name} - Aggregated Confusion Matrix (CV)")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(cm_cv_path, dpi=150)
        plt.close()

        roc_cv_path = os.path.join(PLOTS_DIR, f"roc_cv_{model_name}.png")
        plt.figure(figsize=(7, 6))
        for i in range(N_SPLITS_CV):
            plt.plot(all_fpr[i], all_tpr[i], alpha=0.25)

        mean_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(mean_fpr)
        for i in range(N_SPLITS_CV):
            mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
        mean_tpr /= N_SPLITS_CV
        mean_auc = auc(mean_fpr, mean_tpr)

        plt.plot(mean_fpr, mean_tpr, linewidth=2, label=f"Mean ROC (AUC = {mean_auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.title(f"{model_name} - ROC Curve (CV)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(roc_cv_path, dpi=150)
        plt.close()

        vprint(f"[{model_name}] Saved plots: {history_path}, {cm_cv_path}, {roc_cv_path}", flush=True)

        # -------- Final train on full dev, evaluate on shared holdout --------
        vprint(f"[{model_name}] Training FINAL model on full DEV, evaluating HOLDOUT...", flush=True)
        vectorizer_final = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM)
        X_dev_vec = vectorizer_final.fit_transform(X_dev)
        X_test_vec = vectorizer_final.transform(X_test)

        final_model = clone(base_model)

        t0 = time.time()
        final_model.fit(X_dev_vec, y_dev)
        final_train_time = time.time() - t0

        t1 = time.time()
        y_test_pred = final_model.predict(X_test_vec)
        y_test_prob = get_scores(final_model, X_test_vec)
        final_eval_time = time.time() - t1

        test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

        holdout_pred_csv = os.path.join(model_dir, "holdout_predictions.csv")
        save_predictions_csv(holdout_pred_csv, X_test, y_test, y_test_pred, y_test_prob)

        holdout_cc_path = os.path.join(model_dir, "holdout_confusion_counts.json")
        with open(holdout_cc_path, "w", encoding="utf-8") as f:
            json.dump({k: test_metrics[k] for k in ["TN","FP","FN","TP"]}, f, ensure_ascii=False, indent=2)

        holdout_metrics_path = os.path.join(model_dir, "holdout_test_metrics.json")
        with open(holdout_metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_name,
                "run_tag": RUN_TAG,
                "holdout_test": {k: test_metrics[k] for k in ["accuracy","precision","recall","f1","roc_auc"]},
                "confusion_counts": {k: test_metrics[k] for k in ["TN","FP","FN","TP"]},
                "final_train_time_sec": float(final_train_time),
                "final_eval_time_sec": float(final_eval_time),
            }, f, ensure_ascii=False, indent=2)

        # Holdout plots
        cm_h = confusion_matrix(y_test, y_test_pred)
        cm_hold_path = os.path.join(PLOTS_DIR, f"cm_holdout_{model_name}.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_h.astype(int), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        plt.title(f"{model_name} - Confusion Matrix (Holdout)")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(cm_hold_path, dpi=150)
        plt.close()

        fpr_h, tpr_h, _ = roc_curve(y_test, y_test_prob)
        auc_h = auc(fpr_h, tpr_h)
        roc_hold_path = os.path.join(PLOTS_DIR, f"roc_holdout_{model_name}.png")
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_h, tpr_h, label=f"AUC = {auc_h:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"{model_name} - ROC Curve (Holdout)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_hold_path, dpi=150)
        plt.close()

        vprint(
            f"[{model_name}] HOLDOUT DONE | "
            f"ACC={test_metrics['accuracy']:.4f} P={test_metrics['precision']:.4f} "
            f"R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} AUC={test_metrics['roc_auc']:.4f} | "
            f"TN={test_metrics['TN']} FP={test_metrics['FP']} FN={test_metrics['FN']} TP={test_metrics['TP']} | "
            f"Train={format_mmss(final_train_time)} Eval={format_mmss(final_eval_time)}"
        )
        vprint(f"[{model_name}] Saved holdout predictions: {holdout_pred_csv}")
        vprint(f"[{model_name}] Saved holdout confusion counts: {holdout_cc_path}")
        vprint(f"[{model_name}] Saved holdout metrics: {holdout_metrics_path}")
        vprint(f"[{model_name}] Saved holdout plots: {cm_hold_path}, {roc_hold_path}")

        # Summary row
        summary_rows.append({
            "Model": model_name,
            "CV_Accuracy_Mean": agg["accuracy"]["mean"],
            "CV_Accuracy_Std": agg["accuracy"]["std"],
            "CV_Precision_Mean": agg["precision"]["mean"],
            "CV_Precision_Std": agg["precision"]["std"],
            "CV_Recall_Mean": agg["recall"]["mean"],
            "CV_Recall_Std": agg["recall"]["std"],
            "CV_F1_Mean": agg["f1"]["mean"],
            "CV_F1_Std": agg["f1"]["std"],
            "CV_ROCAUC_Mean": agg["roc_auc"]["mean"],
            "CV_ROCAUC_Std": agg["roc_auc"]["std"],
            "Holdout_Accuracy": test_metrics["accuracy"],
            "Holdout_Precision": test_metrics["precision"],
            "Holdout_Recall": test_metrics["recall"],
            "Holdout_F1": test_metrics["f1"],
            "Holdout_ROCAUC": test_metrics["roc_auc"],
            "Holdout_TN": test_metrics["TN"],
            "Holdout_FP": test_metrics["FP"],
            "Holdout_FN": test_metrics["FN"],
            "Holdout_TP": test_metrics["TP"],
            "CV_Train_Time_Mean_s": agg["train_time_sec"]["mean"],
            "CV_Eval_Time_Mean_s": agg["eval_time_sec"]["mean"],
            "Holdout_Train_Time_s": final_train_time,
            "Holdout_Eval_Time_s": final_eval_time,
        })

        summary_json[model_name] = {
            "cv": agg,
            "holdout_test": test_metrics,
            "paths": {
                "model_dir": model_dir,
                "cv_summary_json": cv_summary_path,
                "holdout_predictions_csv": holdout_pred_csv,
            }
        }

    except Exception as e:
        vprint(f"\n!!! ERROR while running model {model_name}: {repr(e)}")
        vprint("Continuing to next model...\n")
        continue

# ---------------- SAVE GLOBAL SUMMARY ----------------
vprint("\nSaving GLOBAL summary...", flush=True)
summary_df = pd.DataFrame(summary_rows).sort_values(by="CV_F1_Mean", ascending=False).reset_index(drop=True)

summary_csv_path = os.path.join(RUN_DIR, "results_ml_cv_summary.csv")
summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

summary_json_path = os.path.join(RUN_DIR, "results_ml_cv_summary.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "run_tag": RUN_TAG,
        "config": {
            "random_state": RANDOM_STATE,
            "holdout_test_size": HOLDOUT_TEST_SIZE,
            "n_splits_cv": N_SPLITS_CV,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_ngram": TFIDF_NGRAM,
        },
        "models": summary_json
    }, f, ensure_ascii=False, indent=2)

vprint("============================================================")
vprint("FINAL ML SUMMARY (Unified splits)")
vprint("============================================================")
if len(summary_df) > 0:
    vprint(summary_df.to_string(index=False))
else:
    vprint("No rows in summary_df (unexpected). Check errors above.")

vprint(f"\nSaved summary CSV to: {summary_csv_path}")
vprint(f"Saved summary JSON to: {summary_json_path}")
vprint(f"Plots saved to: {PLOTS_DIR}")
vprint("✓ Done.\n")