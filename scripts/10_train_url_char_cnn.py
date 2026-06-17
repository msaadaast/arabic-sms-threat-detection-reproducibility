import os
import time
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "/home/mohamed/SMS_Project/data/Smishing_Dataset_Final.csv"
OUTPUT_DIR = "/home/mohamed/SMS_Project/smishing_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_CharCNN_detailed.xlsx")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_CharCNN_summary.csv")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_URL_CharCNN_summary.json")

TEXT_COL = "url"
LABEL_COL = "label"

RANDOM_STATE = 42
SPLIT_FILE = os.path.join(OUTPUT_DIR, "url_shared_splits_holdout10_cv5.npz")

# Char-CNN settings
MAX_SEQ_LEN = 200
EMBED_DIM = 32
CONV_FILTERS = 128
KERNEL_SIZE = 5
DENSE_UNITS = 64
DROPOUT_RATE = 0.30

BATCH_SIZE = 64
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3

# ============================================================
# Setup folders and seeds
# ============================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

print("=" * 60)
print("URL Char-CNN Script starting...")
print(f"DATA_PATH  : {DATA_PATH}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"PLOTS_DIR  : {PLOTS_DIR}")
print("=" * 60)

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
# Plot helpers
# ============================================================
def save_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Smishing URL Char-CNN - Holdout Confusion Matrix")
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
    ax.set_title("Smishing URL CNN - Holdout ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Model builder
# ============================================================
def build_vectorizer(train_texts):
    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="character",
        output_mode="int",
        output_sequence_length=MAX_SEQ_LEN,
    )
    train_ds = tf.data.Dataset.from_tensor_slices(train_texts).batch(256)
    vectorizer.adapt(train_ds)
    return vectorizer

def build_model(vectorizer):
    vocab_size = len(vectorizer.get_vocabulary())

    inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="url")
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        mask_zero=False,
    )(x)
    x = tf.keras.layers.Conv1D(
        filters=CONV_FILTERS,
        kernel_size=KERNEL_SIZE,
        activation="relu",
        padding="same",
    )(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

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
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
df[LABEL_COL] = df[LABEL_COL].apply(normalize_label)

X_all = df[TEXT_COL].values.astype(str)
y_all = df[LABEL_COL].values.astype(int)

print(f"Total samples after cleaning: {len(df)}")
print("Label distribution:", dict(pd.Series(y_all).value_counts().sort_index()))

# ============================================================
# Shared split usage
# ============================================================
dev_idx, holdout_idx, folds = load_existing_shared_splits(y_all)

X_dev = X_all[dev_idx]
X_holdout = X_all[holdout_idx]
y_dev = y_all[dev_idx]
y_holdout = y_all[holdout_idx]

print(f"Development set: {len(X_dev)}")
print(f"Holdout set    : {len(X_holdout)}")

# ============================================================
# CV evaluation
# ============================================================
model_name = "Char_CNN"

summary_rows = []
detailed_holdout_rows = []

cv_acc = []
cv_prec = []
cv_rec = []
cv_f1 = []
cv_auc = []

print("\n" + "=" * 60)
print("Running CV + Holdout evaluation...")
print("=" * 60)

cv_start = time.time()

for fold_num, (train_idx, val_idx) in enumerate(folds, start=1):
    print(f"\n--- CV Fold {fold_num}/{len(folds)} ---")

    X_train_fold = X_all[train_idx]
    y_train_fold = y_all[train_idx]
    X_val_fold = X_all[val_idx]
    y_val_fold = y_all[val_idx]

    vectorizer = build_vectorizer(X_train_fold)
    model = build_model(vectorizer)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    model.fit(
        X_train_fold,
        y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
    )

    y_val_score = model.predict(X_val_fold, verbose=0).ravel()
    y_val_pred = (y_val_score >= 0.5).astype(int)

    cv_acc.append(accuracy_score(y_val_fold, y_val_pred))
    cv_prec.append(precision_score(y_val_fold, y_val_pred, zero_division=0))
    cv_rec.append(recall_score(y_val_fold, y_val_pred, zero_division=0))
    cv_f1.append(f1_score(y_val_fold, y_val_pred, zero_division=0))
    cv_auc.append(roc_auc_score(y_val_fold, y_val_score))

    tf.keras.backend.clear_session()

cv_total_time = time.time() - cv_start

# ============================================================
# Final holdout training
# ============================================================
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=y_dev,
)

vectorizer = build_vectorizer(X_subtrain)
model = build_model(vectorizer)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )
]

train_start = time.time()
model.fit(
    X_subtrain,
    y_subtrain,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=callbacks,
)
train_time = time.time() - train_start

# ============================================================
# Holdout prediction
# ============================================================
eval_start = time.time()
y_score = model.predict(X_holdout, verbose=0).ravel()
y_pred = (y_score >= 0.5).astype(int)
eval_time = time.time() - eval_start

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

save_confusion_matrix(confusion_matrix(y_holdout, y_pred), model_name, cm_plot)
save_roc_curve(y_holdout, y_score, model_name, roc_plot)

# ============================================================
# Save outputs
# ============================================================
summary_df = pd.DataFrame([{
    "Model": model_name,
    "CV_Accuracy_Mean": np.mean(cv_acc),
    "CV_Accuracy_Std": np.std(cv_acc),
    "CV_Precision_Mean": np.mean(cv_prec),
    "CV_Precision_Std": np.std(cv_prec),
    "CV_Recall_Mean": np.mean(cv_rec),
    "CV_Recall_Std": np.std(cv_rec),
    "CV_F1_Mean": np.mean(cv_f1),
    "CV_F1_Std": np.std(cv_f1),
    "CV_ROCAUC_Mean": np.mean(cv_auc),
    "CV_ROCAUC_Std": np.std(cv_auc),
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
    "Max_Seq_Len": MAX_SEQ_LEN,
    "Embed_Dim": EMBED_DIM,
    "Conv_Filters": CONV_FILTERS,
    "Kernel_Size": KERNEL_SIZE,
    "Dense_Units": DENSE_UNITS,
    "Batch_Size": BATCH_SIZE,
    "Epochs_Max": EPOCHS,
}])

details_df = pd.DataFrame({
    "url": X_holdout,
    "true_label": y_holdout,
    "pred_label": y_pred,
    "pred_score_phishing": y_score,
})

summary_df.to_csv(SUMMARY_OUTPUT, index=False)
summary_df.to_json(JSON_OUTPUT, orient="records", indent=2)

with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    details_df.to_excel(writer, sheet_name="Holdout_Details", index=False)
    df.to_excel(writer, sheet_name="Cleaned_Dataset", index=False)

print("\n" + "=" * 60)
print("FINAL URL CHAR-CNN SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

print("\nSaved files:")
print(f"- Summary CSV : {SUMMARY_OUTPUT}")
print(f"- Summary JSON: {JSON_OUTPUT}")
print(f"- Excel file  : {EXCEL_OUTPUT}")
print(f"- Plots dir   : {PLOTS_DIR}")