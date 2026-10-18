from __future__ import annotations

import argparse
import hashlib
import json
import string
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.frequent_patterns import apriori


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
PAPER_DATA_PATH = RAW_DIR / "sms-spam-collection-paper-5572.tsv"
SUPPORT_VALUES = [0.002, 0.003, 0.004, 0.0045]
MIN_ALLCONF = 0.13
MIN_FREQUENCY = 0.048
PERCENTAGES = [5, 6, 8, 10, 20, 30, 40]
RANDOM_STATE = 8
EXPECTED_FEATURES = 543
MAX_DF = 83


def ensure_nltk_data() -> None:
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Gagal mengunduh resource NLTK: {package}")


def ensure_dataset() -> Path:
    if not PAPER_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {PAPER_DATA_PATH}")
    return PAPER_DATA_PATH


def load_dataset() -> pd.DataFrame:
    dataset_path = ensure_dataset()
    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if "\t" not in line:
                continue
            label, message = line.split("\t", 1)
            if label in {"ham", "spam"}:
                rows.append((label, message))
    df = pd.DataFrame(rows, columns=["label", "message"])
    if len(df) != 5572:
        raise ValueError(
            f"Dataset versi paper harus berisi 5572 baris; terbaca {len(df)}. "
            "Periksa file dataset TSV lokal."
        )
    counts = df["label"].value_counts().to_dict()
    if counts != {"ham": 4825, "spam": 747}:
        raise ValueError(f"Distribusi label berbeda dari paper: {counts}")
    return df


def wordnet_pos(tag: str) -> str:
    initial = tag[:1]
    mapping = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }
    return mapping.get(initial, wordnet.NOUN)


def preprocess_text(text: str, english_stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    text = text.lower().replace("'s", "")
    text = " ".join(word for word in text.split() if word not in english_stop_words)
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    words = [word for word in text.split() if word.isalpha()]
    tagged_words = pos_tag(words)
    return " ".join(lemmatizer.lemmatize(word, wordnet_pos(tag)) for word, tag in tagged_words)


def make_model(name: str):
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=1)
    return LogisticRegression(random_state=1)


def metrics_for(model, X_test, y_test, selected_count: int, total_count: int) -> dict:
    predictions = model.predict(X_test)
    spam_index = list(model.classes_).index("spam")
    probabilities = model.predict_proba(X_test)[:, spam_index]
    return {
        "akurasi": accuracy_score(y_test, predictions),
        "presisi_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f_measure_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "auc": roc_auc_score((y_test == "spam").astype(int), probabilities),
        "presisi_spam": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
        "recall_spam": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
        "f1_spam": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "jumlah_fitur": selected_count,
        "persen_fitur": selected_count / total_count,
    }


def class_transactions(X_train, y_train, feature_names: np.ndarray) -> dict[str, list[list[str]]]:
    transactions: dict[str, list[list[str]]] = {"ham": [], "spam": []}
    matrix = X_train.toarray()
    for row, label in zip(matrix, y_train):
        active = np.flatnonzero(row > 0)
        transactions[label].append(feature_names[active].tolist())
    return transactions


def mine_class_features(transactions: list[list[str]], min_support: float) -> tuple[set[str], dict]:
    if not transactions or not any(transactions):
        return set(), {"transaksi": len(transactions), "itemset_frequent": 0, "itemset_pruned": 0}
    from mlxtend.preprocessing import TransactionEncoder

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    basket = pd.DataFrame(encoded, columns=encoder.columns_)
    frequent = apriori(basket, min_support=min_support, use_colnames=True, low_memory=True)
    if frequent.empty:
        return set(), {"transaksi": len(transactions), "itemset_frequent": 0, "itemset_pruned": 0}

    single_support = {
        next(iter(itemset)): float(support)
        for itemset, support in zip(frequent[frequent["itemsets"].map(len) == 1]["itemsets"], frequent[frequent["itemsets"].map(len) == 1]["support"])
    }
    multi = frequent[frequent["itemsets"].map(len) >= 2].copy()
    if multi.empty:
        return set(), {
            "transaksi": len(transactions), "itemset_frequent": int(len(frequent)),
            "itemset_multi_item": 0, "itemset_pruned": 0, "kata_dalam_itemset": 0,
            "fitur_terpilih": 0,
        }
    multi["all_confidence"] = [
        float(support_value) / max(single_support[item] for item in itemset)
        for itemset, support_value in zip(multi["itemsets"], multi["support"])
    ]
    multi = multi[multi["all_confidence"] > MIN_ALLCONF].copy()

    itemsets = list(multi["itemsets"])
    all_confidences = list(multi["all_confidence"])
    keep = []
    for i, itemset in enumerate(itemsets):
        redundant = False
        for j, larger in enumerate(itemsets):
            if len(itemset) < len(larger) and itemset.issubset(larger):
                if all_confidences[i] <= all_confidences[j]:
                    redundant = True
                    break
        if not redundant:
            keep.append(itemset)

    retained = set(keep)
    counts = Counter(word for itemset in keep for word in itemset)
    unique_count = len(counts)
    selected = {
        word for word, count in counts.items()
        if unique_count and (count / unique_count) > MIN_FREQUENCY
    }
    stats = {
        "transaksi": len(transactions),
        "itemset_frequent": int(len(frequent)),
        "itemset_multi_item": int(len(multi)),
        "itemset_pruned": len(retained),
        "kata_dalam_itemset": unique_count,
        "fitur_terpilih": len(selected),
    }
    return selected, stats


def evaluate_one(name: str, X_train, y_train, X_test, y_test, count: int, total: int) -> dict:
    model = make_model(name)
    model.fit(X_train, y_train)
    row = metrics_for(model, X_test, y_test, count, total)
    row["klasifier"] = name
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduksi paper seleksi fitur SMS spam.")
    parser.add_argument(
        "--support", type=float, choices=SUPPORT_VALUES, default=None,
        help="Jalankan hanya satu nilai min_support; default menjalankan keempat nilai paper.",
    )
    args = parser.parse_args()
    ensure_nltk_data()
    dataset = load_dataset()
    english_stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    dataset["clean_text"] = dataset["message"].map(
        lambda text: preprocess_text(text, english_stop_words, lemmatizer)
    )

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        dataset["clean_text"], dataset["label"], test_size=0.30, random_state=RANDOM_STATE
    )
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    vectorizer = TfidfVectorizer(
        encoding="utf-8", ngram_range=(1, 1), lowercase=False,
        min_df=10, max_df=MAX_DF, norm="l2", sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    feature_names = vectorizer.get_feature_names_out()
    total_features = len(feature_names)
    if total_features != EXPECTED_FEATURES:
        print(
            f"Catatan: paper melaporkan {EXPECTED_FEATURES} fitur; run ini menghasilkan {total_features}."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "initial_features.json").write_text(
        json.dumps(feature_names.tolist(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (RESULTS_DIR / "dataset_summary.json").write_text(
        json.dumps({
            "jumlah_pesan": len(dataset),
            "jumlah_ham": int((dataset["label"] == "ham").sum()),
            "jumlah_spam": int((dataset["label"] == "spam").sum()),
            "train": len(y_train), "test": len(y_test),
            "dataset_file": PAPER_DATA_PATH.name,
            "dataset_sha256": hashlib.sha256(PAPER_DATA_PATH.read_bytes()).hexdigest(),
            "fitur_tfidf_train": total_features,
            "fitur_dilaporkan_paper": EXPECTED_FEATURES,
            "max_df_dipakai": MAX_DF,
            "pesan": "Snapshot 5572; mengecualikan dua pesan ham yang ditambahkan pada arsip UCI versi lebih baru.",
            "label_train": {key: int(value) for key, value in y_train.value_counts().items()},
            "label_test": {key: int(value) for key, value in y_test.value_counts().items()},
            "random_state": RANDOM_STATE,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    models = ["Naive Bayes", "Decision Tree", "Logistic Regression"]
    for model_name in models:
        rows.append(evaluate_one(model_name, X_train.toarray(), y_train, X_test.toarray(), y_test,
                                 total_features, total_features) | {"skenario": "Tanpa seleksi"})

    for score_function, method_name in [(chi2, "Chi-square"), (f_classif, "ANOVA")]:
        for percentage in PERCENTAGES:
            selector = SelectPercentile(score_func=score_function, percentile=percentage)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_count = X_train_selected.shape[1]
            for model_name in models:
                rows.append(evaluate_one(
                    model_name, X_train_selected.toarray(), y_train,
                    X_test_selected.toarray(), y_test, selected_count, total_features
                ) | {"skenario": method_name, "parameter": percentage})

    supports = [args.support] if args.support is not None else SUPPORT_VALUES
    transactions_by_class = class_transactions(X_train, y_train.to_numpy(), feature_names)
    feature_details = []
    for support in supports:
        selected_union: set[str] = set()
        class_stats = {}
        for label in ["ham", "spam"]:
            class_support = support * len(y_train) / len(transactions_by_class[label])
            selected, stats = mine_class_features(transactions_by_class[label], class_support)
            selected_union.update(selected)
            class_stats[label] = stats
        selected_indices = [i for i, word in enumerate(feature_names) if word in selected_union]
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        for model_name in models:
            rows.append(evaluate_one(
                model_name, X_train_selected.toarray(), y_train,
                X_test_selected.toarray(), y_test, len(selected_indices), total_features
            ) | {"skenario": "Usulan paper", "parameter": support})
        feature_details.append({
            "min_support": support,
            "fitur_union": len(selected_indices),
            "statistik_per_kelas": class_stats,
            "fitur_terpilih": sorted(selected_union),
        })

    result_df = pd.DataFrame(rows)
    result_df["akurasi_persen"] = result_df["akurasi"] * 100
    result_df["presisi_weighted_persen"] = result_df["presisi_weighted"] * 100
    result_df["recall_weighted_persen"] = result_df["recall_weighted"] * 100
    result_df["f_measure_weighted_persen"] = result_df["f_measure_weighted"] * 100
    result_df["auc_persen"] = result_df["auc"] * 100
    result_df["presisi_spam_persen"] = result_df["presisi_spam"] * 100
    result_df["recall_spam_persen"] = result_df["recall_spam"] * 100
    result_df["f1_spam_persen"] = result_df["f1_spam"] * 100
    result_df["f1_macro_persen"] = result_df["f1_macro"] * 100
    result_df.to_csv(RESULTS_DIR / "metrics.csv", index=False, float_format="%.6f")
    (RESULTS_DIR / "proposed_features.json").write_text(
        json.dumps(feature_details, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    paper_summary = result_df[result_df["skenario"] == "Usulan paper"].copy()
    best = paper_summary.sort_values("akurasi", ascending=False).groupby("klasifier", as_index=False).first()
    print(f"Dataset: {len(dataset)} SMS; train={len(y_train)}, test={len(y_test)}; fitur={total_features}")
    print("\nHasil terbaik metode usulan per klasifier:")
    print(best[["parameter", "klasifier", "jumlah_fitur", "akurasi_persen", "presisi_weighted_persen", "recall_weighted_persen", "f_measure_weighted_persen", "auc_persen"]].to_string(index=False))
    print(f"\nHasil lengkap: {RESULTS_DIR / 'metrics.csv'}")

if __name__ == "__main__":
    main()
