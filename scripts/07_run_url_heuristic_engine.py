
import os
import re
import ssl
import math
import json
import socket
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import tldextract

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG / PATHS
# ==========================================================
DATA_PATH = "/home/mohamed/SMS_Project/data/Smishing_Dataset_Final.csv"   # columns: label, url
OUTPUT_DIR = "/home/mohamed/SMS_Project/smishing_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

SPLIT_FILE = os.path.join(OUTPUT_DIR, "url_shared_splits_holdout10_cv5.npz")
DYNAMIC_CACHE_FILE = os.path.join(OUTPUT_DIR, "url_dynamic_cache.csv")
EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_heuristic_detailed.xlsx")
THRESHOLD_OUTPUT = os.path.join(OUTPUT_DIR, "Smishing_heuristic_threshold_summary.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "Smishing_heuristic_summary.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

VERBOSE = True

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.10
CV_FOLDS = 5
THRESHOLD_CANDIDATES = [10, 15, 20, 25, 30, 35, 40]

# ---------- TIMEOUTS ----------
HTTP_TIMEOUT = 2
SSL_TIMEOUT = 3
DNS_TIMEOUT = 3

# ==========================================================
# STATIC SCORING WEIGHTS
# ==========================================================
URL_LENGTH_SCORE = 15
LONG_URL_THRESHOLD = 60

SUBDOMAIN_SCORE = 10
SUBDOMAIN_THRESHOLD = 2

HYPHEN_SCORE = 10
HYPHEN_THRESHOLD = 2

SPECIAL_CHAR_SCORE = 10
SUSPICIOUS_WORD_SCORE = 15

NON_WHITELIST_TLD_SCORE = 10
SHORTENER_PENALTY = 5
HTTP_INSECURE_SCORE = 5

PATH_KEYWORD_SCORE = 10

# ==========================================================
# DYNAMIC SCORING WEIGHTS
# ==========================================================
DNS_FAIL_SCORE = 15
SSL_INVALID_SCORE = 15
HTTP_ERROR_SCORE = 5

# ==========================================================
# LOOKUPS / LISTS
# ==========================================================
extractor = tldextract.TLDExtract(suffix_list_urls=None)

PATH_KEYWORDS = [
    "login", "signin", "sign-in", "account", "bank", "verify", "update",
    "reset", "confirm", "secure", "wallet", "pay", "payment", "otp",
    "credentials", "auth", "id", "dashboard", "checkout"
]

SUSPICIOUS_WORDS = [
    "verify", "update", "login", "bank", "reset", "confirm",
    "secure", "unlock", "account", "free", "winner", "gift",
    "urgent", "warning", "password", "reward", "rewards", "money",
    "admin", "otp", "payment", "wallet", "claim", "redeem"
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rebrand.ly", "bit.do", "onelink.to",
    "rb.gy", "shorturl.at", "wa.link", "t.ly", "2u.pw",
    "tiny.cc", "bitly.ws", "shr.mx"
}

COMMON_BENIGN_TLDS = {
    "com", "org", "net", "edu", "gov", "co", "io", "app", "info",
    "biz", "me", "tv", "cc", "eg", "com.eg", "org.eg", "edu.eg",
    "gov.eg", "co.uk", "org.uk", "edu.au", "org.au", "net.au",
    "ae", "sa", "qa", "bh", "kw", "om", "jo", "ma", "tn", "us",
    "uk", "de", "fr", "jp", "au"
}


SUSPICIOUS_SPECIAL_CHARS = ["@", ";", "{", "}", "[", "]", "|", "\\"]


# ==========================================================
# HELPERS
# ==========================================================
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

def normalize_url(url: str) -> str:
    url = str(url).strip().rstrip(".,:;!?")
    if url.lower().startswith("www."):
        return "http://" + url
    return url

def get_domain_parts(url: str):
    ext = extractor(url)
    suffix = ext.suffix.lower()
    domain = ext.domain.lower()
    subdomain = ext.subdomain.lower()
    full_domain = f"{domain}.{suffix}" if domain and suffix else domain
    return full_domain, domain, suffix, subdomain

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def parse_url_safe(url: str):
    url = normalize_url(url)
    return urlparse(url)

def get_host(url: str) -> str:
    parsed = parse_url_safe(url)
    host = parsed.netloc.lower().strip()
    if ":" in host:
        host = host.split(":")[0]
    return host

def get_path_query(url: str):
    parsed = parse_url_safe(url)
    path = parsed.path or ""
    query = parsed.query or ""
    return path, query

def get_final_tld(suffix: str) -> str:
    if not suffix:
        return ""
    return suffix.split(".")[-1]

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def save_confusion_matrix(cm, title, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    classes = ["Benign (0)", "Phishing (1)"]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_roc_curve(y_true, y_score, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# Rare-class stratification parameters
SMS_NATIVE_PHISHING_RAW_ROWS = (0, 23)        
SMS_NATIVE_HAM_RAW_ROWS      = (4524, 5089)   
SMS_NATIVE_PHISHING_HOLDOUT_SIZE = 0.40       



def _identify_sms_native_positions_in_cleaned(raw_df, cleaned_df):
    """
    Identify positions of the SMS-native subgroups in the deduplicated dataset.

    The SMS-native URLs are defined by row position in the RAW CSV. Because
    drop_duplicates may remove some of them (if a duplicate exists later in
    the file), we re-identify them by matching URL strings against the
    deduplicated DataFrame.

    Returns:
        sms_native_phishing_positions: array of positions in cleaned_df
        sms_native_ham_positions:      array of positions in cleaned_df
    """
    
    sms_native_phishing_urls = (
        raw_df["url"]
        .astype(str)
        .str.strip()
        .iloc[SMS_NATIVE_PHISHING_RAW_ROWS[0] : SMS_NATIVE_PHISHING_RAW_ROWS[1] + 1]
        .tolist()
    )
    sms_native_ham_urls = (
        raw_df["url"]
        .astype(str)
        .str.strip()
        .iloc[SMS_NATIVE_HAM_RAW_ROWS[0] : SMS_NATIVE_HAM_RAW_ROWS[1] + 1]
        .tolist()
    )

    
    cleaned_url_to_position = {
        url: pos for pos, url in enumerate(cleaned_df["url"].values)
    }

    sms_native_phishing_positions = np.array(
        [cleaned_url_to_position[u] for u in sms_native_phishing_urls
         if u in cleaned_url_to_position],
        dtype=int,
    )
    sms_native_ham_positions = np.array(
        [cleaned_url_to_position[u] for u in sms_native_ham_urls
         if u in cleaned_url_to_position],
        dtype=int,
    )

    return sms_native_phishing_positions, sms_native_ham_positions


def load_or_create_splits(y_all, cleaned_df, raw_df, split_file=SPLIT_FILE):
    """
    Create a rare-class stratified train/dev/holdout split.

    The function ALWAYS regenerates the split (overwrites any existing
    split file) to ensure consistency with the rare-class stratification
    protocol.

    Stratification design:
      - SMS-native phishing URLs (24 rows): 40/60 holdout/dev split
      - All other URL records: 10/90 holdout/dev split
    The two strata are split independently with the same random seed,
    then concatenated to form the final dev/holdout indices.

    CV folds are constructed on the combined development set using
    standard stratified K-Fold on the ham/phishing label.
    """
    y_all = np.asarray(y_all)
    n_samples = len(y_all)

    print("\n--- Building rare-class stratified split ---")
    print(f"Total samples after cleaning   : {n_samples}")

    # Identify SMS-native subgroup positions in the cleaned dataset
    sms_native_phishing_pos, sms_native_ham_pos = (
        _identify_sms_native_positions_in_cleaned(raw_df, cleaned_df)
    )

    print(f"SMS-native phishing URLs found : {len(sms_native_phishing_pos)} / 24")
    print(f"SMS-native ham URLs found      : {len(sms_native_ham_pos)} / 566")

    
    sms_native_phishing_set = set(sms_native_phishing_pos.tolist())
    all_positions = np.arange(n_samples)
    other_positions = np.array(
        [p for p in all_positions if p not in sms_native_phishing_set],
        dtype=int,
    )

    print(f"Other URL records              : {len(other_positions)}")

   
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_native = rng.permutation(sms_native_phishing_pos)
    n_native_holdout = int(round(
        len(sms_native_phishing_pos) * SMS_NATIVE_PHISHING_HOLDOUT_SIZE
    ))
    native_holdout_idx = shuffled_native[:n_native_holdout]
    native_dev_idx     = shuffled_native[n_native_holdout:]

    print(f"\nSMS-native phishing stratum: "
          f"{len(native_dev_idx)} dev / {len(native_holdout_idx)} holdout "
          f"({100 * SMS_NATIVE_PHISHING_HOLDOUT_SIZE:.0f}% holdout)")

   
    other_labels = y_all[other_positions]
    other_dev_local, other_holdout_local = train_test_split(
        np.arange(len(other_positions)),
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        stratify=other_labels,
    )
    other_dev_idx     = other_positions[other_dev_local]
    other_holdout_idx = other_positions[other_holdout_local]

    print(f"Other URLs stratum         : "
          f"{len(other_dev_idx)} dev / {len(other_holdout_idx)} holdout "
          f"({100 * HOLDOUT_SIZE:.0f}% holdout)")

    # Combine strata
    dev_idx     = np.sort(np.concatenate([native_dev_idx,     other_dev_idx]))
    holdout_idx = np.sort(np.concatenate([native_holdout_idx, other_holdout_idx]))

    # Sanity checks
    assert len(set(dev_idx.tolist()) & set(holdout_idx.tolist())) == 0, \
        "Overlap detected between dev and holdout indices."
    assert len(dev_idx) + len(holdout_idx) == n_samples, \
        f"Index count mismatch: dev({len(dev_idx)}) + holdout({len(holdout_idx)}) " \
        f"!= total({n_samples})."

    print(f"\nFinal split:")
    print(f"  Development set : {len(dev_idx)}")
    print(f"  Holdout set     : {len(holdout_idx)}")
    print(f"  Holdout label distribution: "
          f"{dict(pd.Series(y_all[holdout_idx]).value_counts().sort_index())}")
    print(f"  SMS-native phishing in holdout: {len(native_holdout_idx)}")
    print(f"  SMS-native phishing in dev    : {len(native_dev_idx)}")

    # CV folds on the development set using standard stratified K-Fold on label
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    y_dev = y_all[dev_idx]

    for tr_local, va_local in skf.split(dev_idx, y_dev):
        tr_full = dev_idx[tr_local]
        va_full = dev_idx[va_local]
        folds.append((tr_full, va_full))

    # Save (overwrites any existing file)
    np.savez_compressed(
        split_file,
        n_samples=n_samples,
        y_all=y_all,
        dev_idx=dev_idx,
        holdout_idx=holdout_idx,
        folds=np.array(folds, dtype=object),
    )

    print(f"\n✓ Wrote shared split file: {split_file}")
    print(f"  (Any pre-existing file at this path was overwritten.)")

    return dev_idx, holdout_idx, folds, False

# ==========================================================
# DYNAMIC FEATURE CACHE
# ==========================================================
def load_dynamic_cache():
    if os.path.exists(DYNAMIC_CACHE_FILE):
        cache_df = pd.read_csv(DYNAMIC_CACHE_FILE)
        cache_df["url"] = cache_df["url"].astype(str)
        return {
            row["url"]: {
                "dns_resolution_failure": int(row["dns_resolution_failure"]),
                "invalid_ssl": int(row["invalid_ssl"]),
                "http_response_failure": int(row["http_response_failure"]),
            }
            for _, row in cache_df.iterrows()
        }
    return {}

def save_dynamic_cache(cache_dict):
    rows = []
    for url, vals in cache_dict.items():
        rows.append({
            "url": url,
            "dns_resolution_failure": vals["dns_resolution_failure"],
            "invalid_ssl": vals["invalid_ssl"],
            "http_response_failure": vals["http_response_failure"],
        })
    pd.DataFrame(rows).to_csv(DYNAMIC_CACHE_FILE, index=False)

# ==========================================================
# STATIC FEATURE RULES
# ==========================================================
def check_long_url(url: str) -> int:
    return URL_LENGTH_SCORE if len(url) >= LONG_URL_THRESHOLD else 0

def check_subdomains(url: str) -> int:
    _, _, _, subdomain = get_domain_parts(url)
    if subdomain == "":
        return 0
    depth = len([p for p in subdomain.split(".") if p])
    return SUBDOMAIN_SCORE if depth >= SUBDOMAIN_THRESHOLD else 0

def check_hyphens(url: str) -> int:
    return HYPHEN_SCORE if url.count("-") >= HYPHEN_THRESHOLD else 0

def check_special_chars(url: str) -> int:
    return SPECIAL_CHAR_SCORE if any(c in url for c in SUSPICIOUS_SPECIAL_CHARS) else 0

def check_suspicious_words(url: str) -> int:
    u = url.lower()
    return SUSPICIOUS_WORD_SCORE if any(w in u for w in SUSPICIOUS_WORDS) else 0

def check_insecure_http(url: str) -> int:
    return HTTP_INSECURE_SCORE if normalize_url(url).lower().startswith("http://") else 0

def check_tld_and_shortener(url: str) -> int:
    full_domain, _, suffix, _ = get_domain_parts(url)

    if full_domain in SHORTENERS:
        return SHORTENER_PENALTY

    final_tld = get_final_tld(suffix)
    if suffix in COMMON_BENIGN_TLDS or final_tld in COMMON_BENIGN_TLDS:
        return 0

    return NON_WHITELIST_TLD_SCORE

def check_path_keywords(url: str, base_score: int) -> int:
    if base_score < 10:
        return 0

    path, query = get_path_query(url)
    path_query = (path + "?" + query).lower() if query else path.lower()

    if not path_query:
        return 0

    return PATH_KEYWORD_SCORE if any(kw in path_query for kw in PATH_KEYWORDS) else 0

# ==========================================================
# DYNAMIC CHECKS
# ==========================================================
def dns_resolves(domain: str) -> bool:
    try:
        if not domain:
            return False
        socket.setdefaulttimeout(DNS_TIMEOUT)
        socket.gethostbyname(domain)
        return True
    except Exception:
        return False

def ssl_valid_if_https(url: str):
    """
    Returns:
      True  -> HTTPS exists and certificate is valid
      False -> HTTPS exists but certificate/handshake is invalid
      None  -> URL is not HTTPS or no meaningful SSL check should apply
    """
    parsed = parse_url_safe(url)
    scheme = parsed.scheme.lower()
    domain = get_host(url)

    if scheme != "https":
        return None
    if not domain:
        return None

    try:
        sock = socket.create_connection((domain, 443), timeout=SSL_TIMEOUT)
    except Exception:
        return False

    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
            cert = ssock.getpeercert()
            return True if cert else False
    except ssl.SSLError:
        return False
    except Exception:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass

def http_ok(url: str) -> bool:
    """
    HEAD first, then GET fallback for 403/405 or HEAD-specific failures.
    """
    url = normalize_url(url)

    try:
        r = requests.head(url, allow_redirects=True, timeout=HTTP_TIMEOUT)
        if 200 <= r.status_code < 400:
            return True
        if r.status_code in {403, 405}:
            rg = requests.get(url, allow_redirects=True, timeout=HTTP_TIMEOUT, stream=True)
            return True if 200 <= rg.status_code < 400 else False
        return False
    except Exception:
        try:
            rg = requests.get(url, allow_redirects=True, timeout=HTTP_TIMEOUT, stream=True)
            return True if 200 <= rg.status_code < 400 else False
        except Exception:
            return False

# ==========================================================
# SCORE COMPUTATION
# ==========================================================
def compute_url_score(url: str, dynamic_cache: dict) -> dict:
    url = normalize_url(url)

    if VERBOSE:
        print(f" → Evaluating URL: {url}")

    s_long = check_long_url(url)
    s_subd = check_subdomains(url)
    s_hyp = check_hyphens(url)
    s_spec = check_special_chars(url)
    s_susp = check_suspicious_words(url)
    s_http_insec = check_insecure_http(url)
    s_tld = check_tld_and_shortener(url)

    domain = get_host(url)

    if url in dynamic_cache:
        dyn = dynamic_cache[url]
        s_dns = DNS_FAIL_SCORE if dyn["dns_resolution_failure"] else 0
        s_ssl = SSL_INVALID_SCORE if dyn["invalid_ssl"] else 0
        s_http = HTTP_ERROR_SCORE if dyn["http_response_failure"] else 0
    else:
        dns_ok = dns_resolves(domain)
        s_dns = 0 if dns_ok else DNS_FAIL_SCORE

        ssl_status = ssl_valid_if_https(url)
        s_ssl = SSL_INVALID_SCORE if ssl_status is False else 0

        http_ok_flag = http_ok(url)
        s_http = 0 if http_ok_flag else HTTP_ERROR_SCORE

        dynamic_cache[url] = {
            "dns_resolution_failure": int(s_dns > 0),
            "invalid_ssl": int(s_ssl > 0),
            "http_response_failure": int(s_http > 0),
        }

    base_total = (
        s_long + s_subd + s_hyp + s_spec + s_susp +
        s_http_insec + s_tld + s_dns + s_ssl + s_http
    )

    s_path_kw = check_path_keywords(url, base_total)
    total = base_total + s_path_kw

    if VERBOSE:
        print(f"    · Long URL score       : {s_long}")
        print(f"    · Subdomain score      : {s_subd}")
        print(f"    · Hyphens score        : {s_hyp}")
        print(f"    · Special chars score  : {s_spec}")
        print(f"    · Suspicious words     : {s_susp}")
        print(f"    · HTTP insecure score  : {s_http_insec}")
        print(f"    · TLD/Shortener score  : {s_tld}")
        print(f"    · DNS fail score       : {s_dns}")
        print(f"    · SSL invalid score    : {s_ssl}")
        print(f"    · HTTP error score     : {s_http}")
        print(f"    · Path keyword score   : {s_path_kw}")
        print(f"    → TOTAL SCORE          : {total}")

    return {
        "score_total": total,
        "domain": domain,
        "score_long": s_long,
        "score_subdomains": s_subd,
        "score_hyphens": s_hyp,
        "score_special_chars": s_spec,
        "score_suspicious_words": s_susp,
        "score_http_insecure": s_http_insec,
        "score_tld": s_tld,
        "score_path_keywords": s_path_kw,
        "score_dns": s_dns,
        "score_ssl": s_ssl,
        "score_http": s_http,
    }

# ==========================================================
# THRESHOLD SELECTION
# ==========================================================
def evaluate_thresholds(scores, labels, thresholds):
    rows = []
    for th in thresholds:
        preds = (scores >= th).astype(int)
        m = compute_metrics(labels, preds)
        rows.append({
            "threshold": th,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tp": m["tp"],
        })
    return pd.DataFrame(rows)

def choose_best_threshold(dev_threshold_df):

    sorted_df = dev_threshold_df.sort_values(
        by=["f1", "recall", "precision", "fp"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)
    return int(sorted_df.loc[0, "threshold"]), sorted_df

# ==========================================================
# MAIN
# ==========================================================
def main():
    print("\n========= Smishing URL Heuristic Engine v4 =========")
    print("Loading dataset:", DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns or "url" not in df.columns:
        raise ValueError("CSV must contain 'label' and 'url' columns.")

    df["label"] = df["label"].apply(normalize_label)
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]


    raw_df = df.copy().reset_index(drop=True)

    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    print(f"Total rows after cleaning: {len(df)}")
    print("Label distribution:", df["label"].value_counts().sort_index().to_dict())


    dev_idx, holdout_idx, folds, loaded_existing = load_or_create_splits(
        df["label"].values, cleaned_df=df, raw_df=raw_df
    )

    # Dynamic cache
    dynamic_cache = load_dynamic_cache()

    # Compute scores for all URLs once
    extracted_urls = []
    domains = []

    scores_total = []
    scores_long = []
    scores_subdomains = []
    scores_hyphens = []
    scores_special = []
    scores_suspicious = []
    scores_http_insecure = []
    scores_tld = []
    scores_path_keywords = []
    scores_dns = []
    scores_ssl = []
    scores_http = []

    n = len(df)

    for i, row in df.iterrows():
        if VERBOSE:
            print(f"\n[{i+1}/{n}] Processing row...")

        url = normalize_url(row["url"])
        extracted_urls.append(url)

        feature = compute_url_score(url, dynamic_cache)

        scores_total.append(feature["score_total"])
        scores_long.append(feature["score_long"])
        scores_subdomains.append(feature["score_subdomains"])
        scores_hyphens.append(feature["score_hyphens"])
        scores_special.append(feature["score_special_chars"])
        scores_suspicious.append(feature["score_suspicious_words"])
        scores_http_insecure.append(feature["score_http_insecure"])
        scores_tld.append(feature["score_tld"])
        scores_path_keywords.append(feature["score_path_keywords"])
        scores_dns.append(feature["score_dns"])
        scores_ssl.append(feature["score_ssl"])
        scores_http.append(feature["score_http"])
        domains.append(feature["domain"])

    # Save updated dynamic cache
    save_dynamic_cache(dynamic_cache)

    # Attach columns
    df["normalized_url"] = extracted_urls
    df["domain"] = domains
    df["score_total"] = scores_total
    df["score_long"] = scores_long
    df["score_subdomains"] = scores_subdomains
    df["score_hyphens"] = scores_hyphens
    df["score_special_chars"] = scores_special
    df["score_suspicious_words"] = scores_suspicious
    df["score_http_insecure"] = scores_http_insecure
    df["score_tld"] = scores_tld
    df["score_path_keywords"] = scores_path_keywords
    df["score_dns"] = scores_dns
    df["score_ssl"] = scores_ssl
    df["score_http"] = scores_http

    # ------------------------------------------------------
    # Threshold tuning on DEVELOPMENT set only
    # ------------------------------------------------------
    dev_scores = df.loc[dev_idx, "score_total"].values
    dev_labels = df.loc[dev_idx, "label"].values

    dev_threshold_df = evaluate_thresholds(dev_scores, dev_labels, THRESHOLD_CANDIDATES)
    best_threshold, sorted_dev_threshold_df = choose_best_threshold(dev_threshold_df)

    print("\n============================================================")
    print("Development Threshold Sweep")
    print("============================================================")
    print(dev_threshold_df.to_string(index=False))
    print(f"\nSelected threshold from development set: {best_threshold}")

    # ------------------------------------------------------
    # Final evaluation on HOLDOUT
    # ------------------------------------------------------
    holdout_scores = df.loc[holdout_idx, "score_total"].values
    holdout_labels = df.loc[holdout_idx, "label"].values
    holdout_preds = (holdout_scores >= best_threshold).astype(int)

    holdout_metrics = compute_metrics(holdout_labels, holdout_preds)
    cm = confusion_matrix(holdout_labels, holdout_preds)

    df["pred"] = (df["score_total"] >= best_threshold).astype(int)

    print("\n============================================================")
    print("Smishing URL Heuristic Results (Holdout)")
    print("============================================================")
    print(f"Threshold : {best_threshold}")
    print(f"Accuracy  : {holdout_metrics['accuracy']:.4f}")
    print(f"Precision : {holdout_metrics['precision']:.4f}")
    print(f"Recall    : {holdout_metrics['recall']:.4f}")
    print(f"F1 Score  : {holdout_metrics['f1']:.4f}")
    print(f"Confusion : TN={holdout_metrics['tn']}, FP={holdout_metrics['fp']}, FN={holdout_metrics['fn']}, TP={holdout_metrics['tp']}")

    # ROC on holdout using continuous score_total
    fpr, tpr, _ = roc_curve(holdout_labels, holdout_scores)
    roc_auc = auc(fpr, tpr)

    # Save plots
    save_roc_curve(
        holdout_labels,
        holdout_scores,
        "Smishing URL Heuristic - Holdout ROC Curve",
        os.path.join(PLOTS_DIR, "smishing_heuristic_holdout_roc_curve.png")
    )
    save_confusion_matrix(
        cm,
        "Smishing URL Heuristic - Holdout Confusion Matrix",
        os.path.join(PLOTS_DIR, "smishing_heuristic_holdout_confusion_matrix.png")
    )

    # Save outputs
    dev_threshold_df.to_csv(THRESHOLD_OUTPUT, index=False)

    summary_payload = {
        "selected_threshold": best_threshold,
        "holdout_accuracy": holdout_metrics["accuracy"],
        "holdout_precision": holdout_metrics["precision"],
        "holdout_recall": holdout_metrics["recall"],
        "holdout_f1": holdout_metrics["f1"],
        "holdout_roc_auc": roc_auc,
        "holdout_tn": holdout_metrics["tn"],
        "holdout_fp": holdout_metrics["fp"],
        "holdout_fn": holdout_metrics["fn"],
        "holdout_tp": holdout_metrics["tp"],
        "n_total": int(len(df)),
        "n_dev": int(len(dev_idx)),
        "n_holdout": int(len(holdout_idx)),
        "threshold_candidates": THRESHOLD_CANDIDATES,
        "split_file": SPLIT_FILE,
        "dynamic_cache_file": DYNAMIC_CACHE_FILE,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all_scores")
        dev_threshold_df.to_excel(writer, index=False, sheet_name="dev_threshold_sweep")
        pd.DataFrame([summary_payload]).to_excel(writer, index=False, sheet_name="holdout_summary")
        pd.DataFrame({"dev_idx": pd.Series(dev_idx)}).to_excel(writer, index=False, sheet_name="dev_indices")
        pd.DataFrame({"holdout_idx": pd.Series(holdout_idx)}).to_excel(writer, index=False, sheet_name="holdout_indices")

    print("\nSaved files:")
    print(f"- Shared split file : {SPLIT_FILE}")
    print(f"- Dynamic cache     : {DYNAMIC_CACHE_FILE}")
    print(f"- Threshold CSV     : {THRESHOLD_OUTPUT}")
    print(f"- Excel output      : {EXCEL_OUTPUT}")
    print(f"- Summary JSON      : {SUMMARY_JSON}")
    print(f"- Plots dir         : {PLOTS_DIR}")

    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()