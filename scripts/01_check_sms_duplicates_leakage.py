

import os
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = "/home/mohamed/SMS_Project/data/ARA_SMS_Dataset_Final.csv"
OUTPUT_DIR = "/home/mohamed/SMS_Project"

RANDOM_STATE = 42
HOLDOUT_TEST_SIZE = 0.10
N_SPLITS_CV = 5

# Similarity thresholds
THRESHOLD_HIGH = 0.85   
THRESHOLD_LOW = 0.70    

SPLITS_PATH = os.path.join(
    OUTPUT_DIR,
    "splits",
    f"splits_holdout{int(HOLDOUT_TEST_SIZE*100)}_cv{N_SPLITS_CV}_seed{RANDOM_STATE}.npz"
)

REPORT_DIR = os.path.join(OUTPUT_DIR, "duplicate_check")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = AR_DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("[يى]", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(
        r"[^ء-ي0-9A-Za-z\u0660-\u0669\s@#\$%\^\&\*\(\)\-_=+\:;,.?!]",
        " ",
        text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── LOAD DATASET ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Near-Duplicate Analysis — Train vs. Test")
print("=" * 60)

print(f"\n[1/5] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Message", "Label"]).reset_index(drop=True)

df["Label"] = (
    df["Label"].astype(str).str.strip().str.lower()
    .map(lambda x: "spam" if "spam" in x else "ham")
)
df["clean"] = df["Message"].apply(normalize_arabic)
df["label_bin"] = (df["Label"] == "spam").astype(int)

print(f"    Total messages loaded : {len(df):,}")
print(f"    Ham  : {(df['label_bin'] == 0).sum():,}")
print(f"    Spam : {(df['label_bin'] == 1).sum():,}")

# ── LOAD SHARED SPLIT ─────────────────────────────────────────────────────────
print(f"\n[2/5] Loading shared split file: {SPLITS_PATH}")

if not os.path.exists(SPLITS_PATH):
    raise FileNotFoundError(
        f"Split file not found: {SPLITS_PATH}\n"
        "Run any model script first to generate it, then re-run this script."
    )

split_data = np.load(SPLITS_PATH, allow_pickle=True)
dev_idx = split_data["dev_idx"]
holdout_idx = split_data["holdout_idx"]

if len(dev_idx) + len(holdout_idx) != len(df):
    raise ValueError(
        f"Split file covers {len(dev_idx) + len(holdout_idx)} samples "
        f"but dataset has {len(df)}. Re-generate the split file."
    )

X_train = df["clean"].values[dev_idx]       
X_test = df["clean"].values[holdout_idx]   
y_train = df["label_bin"].values[dev_idx]
y_test = df["label_bin"].values[holdout_idx]

print(f"    Train (dev) size : {len(X_train):,}")
print(f"    Test (holdout)   : {len(X_test):,}")
print("    Split confirmed identical to model scripts ✓")

# ── TF-IDF VECTORISATION ──────────────────────────────────────────────────────
print("\n[3/5] Fitting TF-IDF on full corpus (train + test combined)...")

all_texts = np.concatenate([X_train, X_test])

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    max_features=50_000,
    sublinear_tf=True,
)
tfidf_all = vectorizer.fit_transform(all_texts)

tfidf_train = tfidf_all[:len(X_train)]
tfidf_test = tfidf_all[len(X_train):]

print(f"    Vocabulary size  : {len(vectorizer.vocabulary_):,}")
print(f"    Train matrix     : {tfidf_train.shape}")
print(f"    Test  matrix     : {tfidf_test.shape}")

# ── COSINE SIMILARITY ─────────────────────────────────────────────────────────
print("\n[4/5] Computing cosine similarities (test × train)...")
print(
    f"    This computes {len(X_test):,} × {len(X_train):,} = "
    f"{len(X_test) * len(X_train):,} pairs — may take 1–3 minutes..."
)

BATCH_SIZE = 200
max_sim_per_test = np.zeros(len(X_test))
best_match_idx = np.zeros(len(X_test), dtype=int)

for start in range(0, len(X_test), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(X_test))
    batch = tfidf_test[start:end]
    sims = cosine_similarity(batch, tfidf_train)
    max_sim_per_test[start:end] = sims.max(axis=1)
    best_match_idx[start:end] = sims.argmax(axis=1)
    if (start // BATCH_SIZE) % 5 == 0:
        pct = end / len(X_test) * 100
        print(f"    Progress: {end:,}/{len(X_test):,} test messages ({pct:.0f}%)")

print("    Done ✓")

# ── COLLECT FLAGGED PAIRS ─────────────────────────────────────────────────────
print("\n[5/5] Collecting flagged pairs...")

flagged_high = []
flagged_low = []

for i in range(len(X_test)):
    sim = float(max_sim_per_test[i])
    j = int(best_match_idx[i])

    entry = {
        "test_idx": int(holdout_idx[i]),
        "train_idx": int(dev_idx[j]),
        "similarity": round(sim, 4),
        "test_label": df["Label"].iloc[holdout_idx[i]],
        "train_label": df["Label"].iloc[dev_idx[j]],
        "same_label": df["Label"].iloc[holdout_idx[i]] == df["Label"].iloc[dev_idx[j]],
        "test_message": df["Message"].iloc[holdout_idx[i]],
        "train_message": df["Message"].iloc[dev_idx[j]],
        "test_clean": X_test[i],
        "train_clean": X_train[j],
    }

    if sim >= THRESHOLD_HIGH:
        flagged_high.append(entry)
    elif sim >= THRESHOLD_LOW:
        flagged_low.append(entry)

# ── REPORT ────────────────────────────────────────────────────────────────────
n_exact = sum(1 for e in flagged_high if e["similarity"] >= 0.999)
n_near = len(flagged_high) - n_exact
n_lenient = len(flagged_low)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Total test messages          : {len(X_test):,}")
print(f"  Total train messages         : {len(X_train):,}")
print(f"  Total pairs evaluated        : {len(X_test) * len(X_train):,}")
print()
print(f"  Threshold ≥ 0.999 (exact)   : {n_exact:,} pairs")
print(f"  Threshold ≥ 0.85  (strict)  : {len(flagged_high):,} pairs")
print(f"  Threshold ≥ 0.70  (lenient) : {len(flagged_high) + n_lenient:,} pairs")
print()

if len(flagged_high) == 0:
    print("  ✓ CLEAN: No near-duplicate messages detected at threshold 0.85.")
    print("    This confirms the holdout test set contains no content")
    print("    seen during training, supporting the validity of reported metrics.")
else:
    print(f"  ⚠ REVIEW REQUIRED: {len(flagged_high)} near-duplicate pair(s) detected at threshold 0.85.")
    print("    Review duplicate_pairs_strict_supplementary.xlsx for manual inspection.")
    print("    These flagged pairs should be assessed to determine whether they")
    print("    represent true duplicates or legitimately distinct campaign messages.")

print()
print(f"  Max similarity (highest pair)  : {max_sim_per_test.max():.4f}")
print(f"  Mean similarity (all test msgs): {max_sim_per_test.mean():.4f}")
print(f"  Median similarity              : {float(np.median(max_sim_per_test)):.4f}")

# ── SAVE REPORT JSON ──────────────────────────────────────────────────────────
report = {
    "dataset_path": DATA_PATH,
    "splits_path": SPLITS_PATH,
    "total_test_messages": int(len(X_test)),
    "total_train_messages": int(len(X_train)),
    "total_pairs_evaluated": int(len(X_test) * len(X_train)),
    "similarity_threshold_strict": THRESHOLD_HIGH,
    "similarity_threshold_lenient": THRESHOLD_LOW,
    "exact_duplicates_found": int(n_exact),
    "near_duplicates_strict": int(len(flagged_high)),
    "near_duplicates_lenient": int(len(flagged_high) + n_lenient),
    "max_similarity": round(float(max_sim_per_test.max()), 4),
    "mean_similarity": round(float(max_sim_per_test.mean()), 4),
    "median_similarity": round(float(np.median(max_sim_per_test)), 4),
    "verdict": "CLEAN" if len(flagged_high) == 0 else "REVIEW_REQUIRED",
    "flagged_pairs_strict": flagged_high,
    "flagged_pairs_lenient_only": flagged_low,
}

report_json_path = os.path.join(REPORT_DIR, "duplicate_report.json")
with open(report_json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n  Saved report to: {report_json_path}")

# ── SAVE FLAGGED PAIRS FILES ──────────────────────────────────────────────────
all_flagged = [
    {**e, "threshold_group": "strict (≥0.85)"}
    for e in flagged_high
] + [
    {**e, "threshold_group": "lenient (0.70–0.85)"}
    for e in flagged_low
]

if all_flagged:
    df_all = pd.DataFrame(all_flagged).sort_values("similarity", ascending=False)

    # Main raw export
    pairs_csv_path = os.path.join(REPORT_DIR, "duplicate_pairs.csv")
    df_all.to_csv(pairs_csv_path, index=False, encoding="utf-8-sig")
    print(f"  Saved flagged pairs to: {pairs_csv_path}")

    
    if flagged_high:
        df_strict = pd.DataFrame(flagged_high).sort_values("similarity", ascending=False).reset_index(drop=True)
        df_strict.insert(0, "pair_id", [f"P{i+1:02d}" for i in range(len(df_strict))])

        
        df_strict["review_status"] = ""
        df_strict["reason_not_duplicate"] = ""
        df_strict["different_date"] = ""
        df_strict["different_promo_code"] = ""
        df_strict["different_expiry"] = ""
        df_strict["different_phrasing"] = ""
        df_strict["different_transaction_or_service_context"] = ""

        supp_csv_path = os.path.join(REPORT_DIR, "duplicate_pairs_strict_supplementary.csv")
        supp_xlsx_path = os.path.join(REPORT_DIR, "duplicate_pairs_strict_supplementary.xlsx")

        df_strict.to_csv(supp_csv_path, index=False, encoding="utf-8-sig")
        df_strict.to_excel(supp_xlsx_path, index=False)

        print(f"  Saved strict supplementary CSV to: {supp_csv_path}")
        print(f"  Saved strict supplementary XLSX to: {supp_xlsx_path}")
else:
    print("  No flagged pairs to save.")

# ── SIMILARITY DISTRIBUTION PLOT ──────────────────────────────────────────────
plt.figure(figsize=(9, 4))
plt.hist(max_sim_per_test, bins=80, color="#378ADD", edgecolor="white", linewidth=0.3)
plt.axvline(
    THRESHOLD_HIGH,
    color="#E24B4A",
    linewidth=1.5,
    linestyle="--",
    label=f"Strict threshold ({THRESHOLD_HIGH})"
)
plt.axvline(
    THRESHOLD_LOW,
    color="#EF9F27",
    linewidth=1.5,
    linestyle="--",
    label=f"Lenient threshold ({THRESHOLD_LOW})"
)
plt.xlabel("Maximum cosine similarity to any train message", fontsize=11)
plt.ylabel("Number of test messages", fontsize=11)
plt.title("Distribution of Maximum Train-Test Similarity per Test Message", fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()

plot_path = os.path.join(REPORT_DIR, "similarity_distribution.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  Saved similarity plot to: {plot_path}")

# ── PAPER-READY STATEMENT ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PAPER-READY STATEMENT (paste into manuscript)")
print("=" * 60)

if len(flagged_high) == 0:
    print("""
To verify the integrity of the holdout evaluation, a near-duplicate
analysis was conducted by computing pairwise cosine similarity between
all test and training messages using character-level TF-IDF
representations (n-gram range 2–4). No message pairs with similarity
≥ 0.85 were detected across the train/test boundary (maximum observed
similarity: {:.4f}), confirming that the holdout set contains no
content seen during training and that the reported metrics reflect
genuine generalisation performance.
""".format(max_sim_per_test.max()))
else:
    print(f"""
To verify the integrity of the holdout partition following deduplication,
a post-split near-duplicate analysis was conducted by computing pairwise
cosine similarity between all holdout messages and all development-set
messages using character-level TF-IDF representations (n-gram range 2–4).
A total of {len(flagged_high)} train/test pairs exceeded the strict
similarity threshold of {THRESHOLD_HIGH:.2f}, with a maximum observed
similarity of {max_sim_per_test.max():.4f}. These flagged pairs require
manual inspection to determine whether they represent true duplicates or
legitimately distinct messages with naturally recurring campaign structure.
""")

print("=" * 60)
print("Done.")