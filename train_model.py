# ==========================================
# GODI DETECTOR
# Training, Model Comparison & Serialization
# ==========================================
#
# Trains three classifiers on one identical TF-IDF feature matrix and one
# identical stratified split, evaluates each of them, writes the comparison
# artefacts to results/ (CSV, JSON, reports, plots), then deploys the selected
# model as models/godi_news_classifier.pkl.
#
# Run from the project root:  python train_model.py

import datetime
import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # headless rendering; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# --- Fixed configuration: every experiment uses exactly these values --------
DATASET_PATH = "data/processed/news_dataset.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
META_PATH = "models/model_meta.json"
LATEST_MODEL_PATH = "models/godi_news_classifier.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TFIDF_PARAMS = {
    "stop_words": "english",
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
}

TARGET_NAMES = ["Real News", "Fake News"]


def load_prepared_dataset(path=DATASET_PATH):
    """
    Load the cleaned dataset and build the model input column.

    The same text construction (title + body, missing values as empty strings,
    empty rows dropped) is used here and by prepare_data.py, so training never
    sees text the cleaning pipeline would have removed.
    """
    news_data = pd.read_csv(path)
    news_data["title"] = news_data["title"].fillna("")
    news_data["text"] = news_data["text"].fillna("")
    news_data["content"] = (news_data["title"] + " " + news_data["text"]).str.strip()
    news_data = news_data[news_data["content"] != ""].reset_index(drop=True)

    X = news_data["content"]
    y = news_data["label"].astype(int)
    return X, y


def split_dataset(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified hold-out split; the test set is never fitted on."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_vectorizer():
    """The one TF-IDF configuration shared by all three experiments."""
    return TfidfVectorizer(**TFIDF_PARAMS)


def build_pipeline(vectorizer, classifier):
    """
    Wrap a fitted vectorizer and classifier in a single sklearn Pipeline.

    Keeping both steps in the pickle means prediction applies exactly the same
    feature extraction as training — there is no second, drifting vectorizer in
    the application.
    """
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def build_classifiers():
    """The three compared algorithms, each with a fixed random state."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }


def _score(y_true, y_pred, y_score=None):
    """Compute every metric the report needs, including per-class values."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_fake": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None:
        # Only models exposing predict_proba get probability metrics.
        metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
        metrics["log_loss"] = float(log_loss(y_true, y_score, labels=[0, 1]))
    return metrics


def evaluate_model(classifier, X_train_vec, X_test_vec, y_train, y_test):
    """
    Fit one classifier on the shared matrices and evaluate it on the hold-out set.

    Returns:
        (fitted classifier, metrics dict, predictions, predicted probabilities or None)
    """
    started = time.perf_counter()
    classifier.fit(X_train_vec, y_train)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    y_pred = classifier.predict(X_test_vec)
    predict_seconds = time.perf_counter() - started

    y_score = None
    if hasattr(classifier, "predict_proba"):
        # Column 1 is the probability of the positive (fake) class.
        y_score = classifier.predict_proba(X_test_vec)[:, 1]

    metrics = _score(y_test, y_pred, y_score)
    metrics["train_seconds"] = round(train_seconds, 2)
    metrics["predict_seconds"] = round(predict_seconds, 2)
    metrics["provides_probabilities"] = y_score is not None
    return classifier, metrics, y_pred, y_score


def select_model(results):
    """
    Choose the model to deploy among those that output genuine probabilities.

    The app shows a confidence score, so a model that only exposes an
    uncalibrated decision margin (Linear SVM) cannot be used as-is. Among the
    probabilistic models, the highest weighted F1 wins.
    """
    eligible = {
        name: res for name, res in results.items() if res["provides_probabilities"]
    }
    if not eligible:
        raise RuntimeError("No evaluated model provides predict_proba().")
    return max(eligible, key=lambda name: eligible[name]["f1_weighted"])


def _save_confusion_matrix(name, matrix, path):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                fontsize=11,
            )
    ax.set_xticks([0, 1], TARGET_NAMES)
    ax.set_yticks([0, 1], TARGET_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_comparison_chart(results, path):
    """Grouped bar chart of accuracy / weighted precision / recall / F1."""
    names = list(results)
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Precision"),
        ("recall_weighted", "Recall"),
        ("f1_weighted", "F1-score"),
    ]

    x = np.arange(len(names))
    width = 0.8 / len(metric_keys)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, (key, label) in enumerate(metric_keys):
        values = [results[name][key] * 100 for name in names]
        bars = ax.bar(x + index * width, values, width, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    # Accuracy is ~97%, so start at 85 to make the differences legible.
    ax.set_ylim(85, 101)
    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2, names)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model comparison on the same 20% hold-out test set")
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results(results, selection, dataset_stats, vectorizer, output_dir=RESULTS_DIR):
    """Write the comparison table, metrics JSON and plots to results/."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, res in results.items():
        rows.append(
            {
                "model": name,
                "accuracy": round(res["accuracy"], 4),
                "precision_weighted": round(res["precision_weighted"], 4),
                "recall_weighted": round(res["recall_weighted"], 4),
                "f1_weighted": round(res["f1_weighted"], 4),
                "precision_fake": round(res["precision_fake"], 4),
                "recall_fake": round(res["recall_fake"], 4),
                "f1_fake": round(res["f1_fake"], 4),
                "train_seconds": res["train_seconds"],
                "predict_seconds": res["predict_seconds"],
                "brier_score": round(res["brier_score"], 4) if "brier_score" in res else None,
                "log_loss": round(res["log_loss"], 4) if "log_loss" in res else None,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    comparison.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    for name, res in results.items():
        slug = name.lower().replace(" ", "_")
        _save_confusion_matrix(
            name,
            res["confusion_matrix"],
            os.path.join(output_dir, f"confusion_matrix_{slug}.png"),
        )
        with open(
            os.path.join(output_dir, f"classification_report_{slug}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {res['accuracy'] * 100:.2f}%\n")
            f.write(
                f"Train time: {res['train_seconds']}s | "
                f"Predict time: {res['predict_seconds']}s\n\n"
            )
            f.write(
                classification_report(
                    res["y_true"],
                    res["y_pred"],
                    target_names=TARGET_NAMES,
                    digits=4,
                )
            )
            f.write("\nConfusion matrix (rows = true, cols = predicted):\n")
            f.write(f"{np.asarray(res['confusion_matrix'])}\n")

    _save_comparison_chart(results, os.path.join(output_dir, "model_comparison.png"))

    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_stats,
        "vectorizer": {
            "stop_words": TFIDF_PARAMS["stop_words"],
            "max_features": TFIDF_PARAMS["max_features"],
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "min_df": TFIDF_PARAMS["min_df"],
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "selection_rule": (
            "Highest weighted F1 among models that expose genuine predict_proba "
            "probabilities (Linear SVM is excluded because it only outputs an "
            "uncalibrated decision margin)."
        ),
        "selected_model": selection,
        "models": {
            name: {
                key: value
                for key, value in res.items()
                if key not in ("y_true", "y_pred")
            }
            for name, res in results.items()
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return comparison, payload


def save_model(pipeline, metrics, algorithm, comparison):
    """Persist the selected pipeline, a timestamped snapshot, and the metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    versioned_path = f"models/godi_news_classifier_{timestamp}.pkl"
    joblib.dump(pipeline, versioned_path)
    joblib.dump(pipeline, LATEST_MODEL_PATH)

    meta = {
        "trained_at": timestamp,
        "algorithm": algorithm,
        "accuracy": round(metrics["accuracy"] * 100, 4),
        "f1_weighted": round(metrics["f1_weighted"] * 100, 4),
        "precision_weighted": round(metrics["precision_weighted"] * 100, 4),
        "recall_weighted": round(metrics["recall_weighted"] * 100, 4),
        "dataset_size": int(metrics["dataset_size"]),
        "train_size": int(metrics["train_size"]),
        "test_size": int(metrics["test_size"]),
        "versioned_file": versioned_path,
        "vectorizer": {
            "stop_words": TFIDF_PARAMS["stop_words"],
            "max_features": TFIDF_PARAMS["max_features"],
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "min_df": TFIDF_PARAMS["min_df"],
        },
        "comparison": [
            {
                "model": row["model"],
                "accuracy": round(float(row["accuracy"]) * 100, 2),
                "f1_weighted": round(float(row["f1_weighted"]) * 100, 2),
            }
            for row in comparison.to_dict(orient="records")
        ],
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return versioned_path, meta


def main():
    print("Loading dataset...")
    X, y = load_prepared_dataset()
    print(f"Dataset loaded: {len(X):,} articles")
    print(f"Class distribution: {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"\nTraining articles: {len(X_train):,}")
    print(f"Testing articles:  {len(X_test):,}")

    print("\nFitting TF-IDF vectorizer (shared by all three models)...")
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Feature matrix: {X_train_vec.shape[0]:,} x {X_train_vec.shape[1]:,}")

    results = {}
    fitted_classifiers = {}
    for name, classifier in build_classifiers().items():
        print(f"\nTraining {name}...")
        fitted, metrics, y_pred, _ = evaluate_model(
            classifier, X_train_vec, X_test_vec, y_train, y_test
        )
        metrics["y_true"] = y_test.tolist()
        metrics["y_pred"] = list(y_pred)
        results[name] = metrics
        fitted_classifiers[name] = fitted

        print(f"  Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"  Weighted F1: {metrics['f1_weighted'] * 100:.2f}%")
        print(f"  Train time: {metrics['train_seconds']}s")
        print(
            "\n"
            + classification_report(y_test, y_pred, target_names=TARGET_NAMES, digits=4)
        )
        print("Confusion matrix (rows = true, cols = predicted):")
        print(np.asarray(metrics["confusion_matrix"]))

    selection = select_model(results)
    print(f"\nSelected model: {selection}")
    print(
        "Reason: highest weighted F1 among models that return genuine class "
        "probabilities, which the UI needs for its confidence score."
    )

    pipeline = build_pipeline(vectorizer, fitted_classifiers[selection])

    dataset_stats = {
        "total_articles": int(len(X)),
        "train_articles": int(len(X_train)),
        "test_articles": int(len(X_test)),
        "real_articles": int((y == 0).sum()),
        "fake_articles": int((y == 1).sum()),
        "features": int(X_train_vec.shape[1]),
    }
    metrics = results[selection]
    metrics["dataset_size"] = dataset_stats["total_articles"]
    metrics["train_size"] = dataset_stats["train_articles"]
    metrics["test_size"] = dataset_stats["test_articles"]

    comparison, _ = save_results(results, selection, dataset_stats, vectorizer)
    versioned_path, _meta = save_model(pipeline, metrics, selection, comparison)

    print("\nComparison table:")
    print(comparison.to_string(index=False))
    print(f"\nResults written to {os.path.abspath(RESULTS_DIR)}")
    print(f"Versioned model saved to: {versioned_path}")
    print(f"Latest model saved to:    {LATEST_MODEL_PATH}")
    print("Model metadata saved to:  models/model_meta.json")
    print("\n================================")
    print("GODI DETECTOR MODEL SAVED!")
    print("================================")


if __name__ == "__main__":
    main()
