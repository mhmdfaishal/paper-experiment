from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DATA_PATH = ROOT / "data" / "raw" / "sms-spam-collection-5572.tsv"
REFERENCE_DIR = ROOT / "data" / "reference"
FEATURES_PATH = REFERENCE_DIR / "initial_features.json"
FS_FAI_METRICS_PATH = REFERENCE_DIR / "fs_fai_metrics.csv"

OUTER_SEED = 8
VALIDATION_SEED = 42
MIN_DF = 10
MAX_DF = 83

_PREPARED_DATA = None


# checking NLTK resources
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

# load dataset
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    rows = []
    with DATA_PATH.open(encoding="utf-8") as file:
        for line in file:
            label, separator, message = line.rstrip("\r\n").partition("\t")
            if separator and label in {"ham", "spam"}:
                rows.append((label, message))

    dataset = pd.DataFrame(rows, columns=["category", "text"])
    counts = dataset["category"].value_counts().to_dict()
    if len(dataset) != 5572 or counts != {"ham": 4825, "spam": 747}:
        raise ValueError("Dataset harus berisi 5.572 SMS: 4.825 ham dan 747 spam.")
    return dataset


# remove punctuations from text
def remove_punctuations(text: str) -> str:
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    return text


# lemmatizing words
def lemmatize_words(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "J": wordnet.ADJ,
        "R": wordnet.ADV,
    }
    tagged_words = pos_tag(text.split())
    return " ".join(
        lemmatizer.lemmatize(word, wordnet_map.get(tag[0], wordnet.NOUN))
        for word, tag in tagged_words
    )


# preprocessing data
def preprocess(dataset: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()
    data["text"] = dataset["text"]
    data["label_encode"] = LabelEncoder().fit_transform(dataset["category"]).astype(str)
    data["label"] = dataset["category"]

    data["text"] = data["text"].map(lambda text: text.lower())
    data["text"] = data["text"].str.replace("'s", "", regex=False)

    stop_words = set(stopwords.words("english"))
    data["text"] = data["text"].apply(
        lambda text: " ".join(word for word in text.split() if word not in stop_words)
    )
    data["text"] = data["text"].apply(remove_punctuations)
    data["text"] = data["text"].apply(
        lambda text: " ".join(word for word in text.split() if word.isalpha())
    )
    data["text"] = data["text"].apply(lemmatize_words)
    return data


# loading data, create featur TF-IDF and splitting main train-test
def prepare_outer_split():
    global _PREPARED_DATA
    if _PREPARED_DATA is not None:
        return _PREPARED_DATA

    ensure_nltk_data()
    data = preprocess(load_dataset())
    train_text, test_text, y_train, y_test = train_test_split(
        data["text"],
        data["label_encode"],
        test_size=0.30,
        random_state=OUTER_SEED,
    )
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        encoding="utf-8",
        ngram_range=(1, 1),
        lowercase=False,
        max_df=MAX_DF,
        min_df=MIN_DF,
        norm="l2",
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    feature_names = vectorizer.get_feature_names_out()

    if len(feature_names) != 543:
        raise ValueError(f"Jumlah fitur harus 543, tetapi diperoleh {len(feature_names)}.")
    expected_features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    if feature_names.tolist() != expected_features:
        raise ValueError("Daftar fitur berbeda dari reproduksi metode paper.")

    _PREPARED_DATA = (X_train, X_test, y_train, y_test, feature_names, data)
    return _PREPARED_DATA


# splitting training data to train and validation data for fitness evaluation
def make_inner_split(X_train, y_train):
    return train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=VALIDATION_SEED,
        stratify=y_train,
    )


# measuring fitness subset feature based on f1-macro validation and reducting feature
def score_mask(mask, X_train, y_train, X_valid, y_valid):
    selected = np.flatnonzero(mask)
    model = LogisticRegression(random_state=1, max_iter=500)
    model.fit(X_train[:, selected], y_train)
    prediction = model.predict(X_valid[:, selected])
    f1_macro = f1_score(y_valid, prediction, average="macro", zero_division=0)
    feature_reduction = 1.0 - len(selected) / X_train.shape[1]
    fitness = 0.50 * f1_macro + 0.50 * feature_reduction
    return float(fitness), float(f1_macro)


# create classifier model
def create_classifier(name: str):
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=1)
    return LogisticRegression(random_state=1, max_iter=1000)


# evaluate selected features using classifier
def evaluate_selected_features(X_train, y_train, X_test, y_test, selected_indices):
    rows = []
    train_data = X_train[:, selected_indices].toarray()
    test_data = X_test[:, selected_indices].toarray()

    for name in ["Naive Bayes", "Decision Tree", "Logistic Regression"]:
        model = create_classifier(name)
        model.fit(train_data, y_train)
        prediction = model.predict(test_data)
        spam_probability = model.predict_proba(test_data)[:, list(model.classes_).index("1")]
        rows.append({
            "klasifier": name,
            "akurasi": accuracy_score(y_test, prediction),
            "presisi_spam": precision_score(y_test, prediction, pos_label="1", zero_division=0),
            "recall_spam": recall_score(y_test, prediction, pos_label="1", zero_division=0),
            "f1_spam": f1_score(y_test, prediction, pos_label="1", zero_division=0),
            "f1_weighted": f1_score(y_test, prediction, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test, prediction, average="macro", zero_division=0),
            "auc": roc_auc_score((y_test == "1").astype(int), spam_probability),
            "jumlah_fitur": len(selected_indices),
            "total_fitur": X_train.shape[1],
            "persen_fitur": len(selected_indices) / X_train.shape[1],
        })
    return rows


# printing performance metrics classifier
def print_performance(rows) -> None:
    columns = [
        "klasifier", "akurasi", "presisi_spam", "recall_spam",
        "f1_spam", "f1_macro", "auc", "jumlah_fitur",
    ]
    table = pd.DataFrame(rows)[columns].copy()
    for column in columns[1:-1]:
        table[column] = (table[column] * 100).map(lambda value: f"{value:.2f}%")
    table = table.rename(columns={
        "klasifier": "Classifier",
        "akurasi": "Accuracy",
        "presisi_spam": "Precision spam",
        "recall_spam": "Recall spam",
        "f1_spam": "F1 spam",
        "f1_macro": "F1 macro",
        "auc": "AUC",
        "jumlah_fitur": "Fitur",
    })
    print("\nPerformance pada test set:")
    print(table.to_string(index=False))


# saving metrics, configuration, fitness, and selected feature on csv file
def save_run(algorithm, config, history, feature_names, selected_indices,
             model_rows, dataset, X_train, y_train, X_test, y_test):
    frame = pd.DataFrame(model_rows)
    metric_columns = [
        "akurasi", "presisi_spam", "recall_spam", "f1_spam",
        "f1_weighted", "f1_macro", "auc", "persen_fitur",
    ]
    for column in metric_columns:
        frame[f"{column}_persen"] = frame[column] * 100

    run_label = config.get("run_label")
    folder = "pso" if algorithm == "bpso" else algorithm
    output_dir = ROOT / folder / "results"
    if run_label:
        output_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{run_label}" if run_label else ""
    frame.to_csv(
        output_dir / f"{algorithm}{suffix}_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    summary = {
        "algoritma": algorithm,
        "konfigurasi": config,
        "dataset_file": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "jumlah_sms": len(dataset),
        "jumlah_train": len(y_train),
        "jumlah_test": len(y_test),
        "jumlah_fitur_awal": X_train.shape[1],
        "jumlah_fitur_terpilih": len(selected_indices),
        "fitur_terpilih": feature_names[selected_indices].tolist(),
        "history": history,
    }
    (output_dir / f"{algorithm}{suffix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# summarize scores across seeds for each method and classifier
def summarize_experiments(rows):
    frame = pd.DataFrame(rows)
    summary = []
    for (method, classifier), group in frame.groupby(["metode", "klasifier"], sort=False):
        row = {"metode": method, "klasifier": classifier, "n": len(group)}
        for metric in ("akurasi", "f1_macro", "f1_weighted", "auc", "jumlah_fitur", "fitness_validasi"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_best"] = values.min() if metric == "jumlah_fitur" else values.max()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_worst"] = values.max() if metric == "jumlah_fitur" else values.min()
        summary.append(row)
    return pd.DataFrame(summary)


# format best, mean, standard deviation, and worst scores.
def show_range(row, metric, scale=100, digits=2):
    return (
        f"{row[f'{metric}_best']*scale:.{digits}f} / "
        f"{row[f'{metric}_mean']*scale:.{digits}f} ± "
        f"{row[f'{metric}_std']*scale:.{digits}f} / "
        f"{row[f'{metric}_worst']*scale:.{digits}f}"
    )


# print test-score summaries for all methods and classifiers.
def print_experiment_summary(rows):
    summary = summarize_experiments(rows)
    print("\n=== Hasil test per classifier ===")
    print("Skor: terbaik / rata-rata ± SD / terburuk. Fitur: minimum / rata-rata ± SD / maksimum.")
    output = []
    for _, row in summary.iterrows():
        output.append({
            "Metode": row.metode,
            "Classifier": row.klasifier,
            "Akurasi %": show_range(row, "akurasi"),
            "F1 macro %": show_range(row, "f1_macro"),
            "AUC %": show_range(row, "auc"),
            "Fitur": show_range(row, "jumlah_fitur", 1, 1),
        })
    print(pd.DataFrame(output).to_string(index=False))


# compare configurations and select the highest mean validation fitness.
def print_parameter_summary(rows, title):
    summary = summarize_experiments(rows)
    summary = summary[summary.klasifier == "Logistic Regression"]
    print(f"\n=== {title} ===")
    output = []
    for _, row in summary.iterrows():
        output.append({
            "Konfigurasi": row.metode,
            "Fitness validasi mean ± SD": f"{row.fitness_validasi_mean:.4f} ± {row.fitness_validasi_std:.4f}",
            "F1 macro test mean ± SD %": f"{row.f1_macro_mean*100:.2f} ± {row.f1_macro_std*100:.2f}",
            "Fitur mean ± SD": f"{row.jumlah_fitur_mean:.1f} ± {row.jumlah_fitur_std:.1f}",
        })
    print(pd.DataFrame(output).to_string(index=False))
    best = summary.loc[summary.fitness_validasi_mean.idxmax()]
    print(f"Fitness validasi rata-rata tertinggi: {best.metode} ({best.fitness_validasi_mean:.4f}).")


# print mean best-so-far fitness at each evaluation checkpoint
def print_convergence(histories):
    print("\n=== Kurva konvergensi rata-rata ===")
    for method, runs in histories.items():
        checkpoints = [entry["evaluasi_kumulatif"] for entry in runs[0]]
        values = []
        for index in range(len(checkpoints)):
            values.append(float(np.mean([history[index]["fitness_terbaik"] for history in runs])))
        print(f"\n{method}")
        print("evaluasi : " + " ".join(f"{value:>6}" for value in checkpoints))
        print("fitness  : " + " ".join(f"{value:>6.3f}" for value in values))



# save raw run scores and protocol settings for independent algorithm runs.
def save_experiment_results(algorithm, rows, feature_names):
    folder = "pso" if algorithm == "bpso" else "ga"
    output_dir = ROOT / folder / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256("\n".join(feature_names).encode()).hexdigest(),
        "outer_seed": OUTER_SEED,
        "validation_seed": VALIDATION_SEED,
        "min_df": MIN_DF,
        "max_df": MAX_DF,
        "objective": "0.50 * F1-macro-validasi + 0.50 * (1 - proporsi_fitur)",
    }
    frame = pd.DataFrame(rows)
    frame["protokol"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    frame.to_csv(output_dir / "experiment_runs.csv", index=False)
    protocol["seeds"] = sorted(int(seed) for seed in frame.seed.unique())
    protocol["metode"] = frame.metode.unique().tolist()
    protocol["evaluasi_per_run"] = sorted(int(value) for value in frame.evaluasi_fitness.unique())
    (output_dir / "experiment_config.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
