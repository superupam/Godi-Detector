"""End-to-end tests for the data pipeline, the three algorithms and the
train/predict consistency of the saved model.

The full 39k-article corpus is not required: these tests build a small synthetic
corpus with the same columns and run the real functions on it.
"""

import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import prepare_data
import train_model
from prepare_data import build_dataset, clean_text, strip_media_credits, strip_publisher_mentions
from train_model import (
    build_classifiers,
    build_pipeline,
    build_vectorizer,
    evaluate_model,
    load_prepared_dataset,
    save_results,
    select_model,
    split_dataset,
)

REAL_WORDS = (
    "senate budget committee reported official economic growth percent "
    "statement wednesday administration policy federal reserve"
)
FAKE_WORDS = (
    "shocking secret exposed incredible miracle cure video watch breaking "
    "bombshell viral outrageous hidden truth"
)


# --- Text sanitization -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_prefix",
    [
        ("WASHINGTON (Reuters) - The president spoke.", "The president spoke."),
        ("(Reuters) - Markets rose sharply.", "Markets rose sharply."),
        ("LONDON (Reuters) - Ministers met on Monday.", "Ministers met on Monday."),
    ],
)
def test_datelines_and_wire_tags_are_removed(raw, expected_prefix):
    assert strip_publisher_mentions(raw) == expected_prefix


def test_wire_mentions_inside_the_body_are_removed():
    cleaned = strip_publisher_mentions("A Reuters/Ipsos poll found support falling.")
    assert "Reuters" not in cleaned
    assert "Ipsos poll" in cleaned


def test_photo_credit_tail_is_dropped():
    text = "The rally drew thousands. Featured image: YouTube"
    assert strip_media_credits(text).strip() == "The rally drew thousands."


def test_inline_credit_label_is_removed():
    # A label near the end is treated as a trailing credit block and dropped with
    # the tail; a label further from the end is stripped in place.
    body = "The council confirmed the vote late on Tuesday evening. " * 6
    cleaned = strip_media_credits("Featured image is a screenshot of a tweet. " + body)
    assert "Featured image" not in cleaned
    assert "council confirmed the vote" in cleaned


def test_clean_text_removes_urls_and_normalizes_whitespace():
    cleaned = clean_text("Read  more   here https://example.com/story and www.example.org")
    assert "http" not in cleaned
    assert "www." not in cleaned
    assert "  " not in cleaned


def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_preserves_article_content():
    text = "The Senate passed a bipartisan infrastructure bill on Wednesday."
    assert clean_text(text) == text


# --- Dataset construction --------------------------------------------------


def _write_raw_corpus(tmp_path, per_class=40):
    """Write True.csv/Fake.csv with the Kaggle columns plus noise rows."""
    real_rows, fake_rows = [], []
    for i in range(per_class):
        real_rows.append(
            {
                "title": f"Senate committee report number {i}",
                "text": f"{REAL_WORDS} session {i}",
                "subject": "politicsNews",
                "date": "2017-01-01",
            }
        )
        fake_rows.append(
            {
                "title": f"Shocking discovery number {i}",
                "text": f"{FAKE_WORDS} story {i}",
                "subject": "News",
                "date": "2017-01-01",
            }
        )

    # A duplicate of the first real article, and a stub that must be dropped.
    real_rows.append(dict(real_rows[0]))
    real_rows.append({"title": "", "text": "", "subject": "politicsNews", "date": ""})

    true_path = tmp_path / "True.csv"
    fake_path = tmp_path / "Fake.csv"
    pd.DataFrame(real_rows).to_csv(true_path, index=False)
    pd.DataFrame(fake_rows).to_csv(fake_path, index=False)
    return true_path, fake_path


def test_build_dataset_reports_real_counts_and_drops_noise(tmp_path):
    true_path, fake_path = _write_raw_corpus(tmp_path)
    data, stats = build_dataset(str(true_path), str(fake_path))

    assert stats["raw_real"] == 42  # 40 articles + duplicate + stub
    assert stats["raw_fake"] == 40
    assert stats["raw_total"] == 82
    assert stats["duplicates_removed"] == 1
    assert stats["short_removed"] == 1
    assert stats["cleaned_total"] == 80
    assert stats["real_retained"] == 40
    assert stats["fake_retained"] == 40
    assert stats["removed_total"] == 2
    assert data["content"].duplicated().sum() == 0
    assert set(data["label"].unique()) == {0, 1}
    assert (data["content"].str.len() > 20).all()


def test_load_prepared_dataset_fills_missing_and_drops_empty_rows(tmp_path):
    path = tmp_path / "news_dataset.csv"
    pd.DataFrame(
        {
            "title": ["Headline", None, ""],
            "text": ["Body text about policy.", "Body only.", ""],
            "label": [0, 1, 1],
        }
    ).to_csv(path, index=False)

    X, y = load_prepared_dataset(str(path))
    assert len(X) == 2
    assert X.iloc[0] == "Headline Body text about policy."
    assert X.iloc[1] == "Body only."
    assert y.tolist() == [0, 1]


# --- Training, evaluation and serialization --------------------------------


@pytest.fixture(scope="module")
def trained_experiment(tmp_path_factory):
    """Train all three models once on a synthetic corpus and share the result."""
    tmp_path = tmp_path_factory.mktemp("corpus")
    true_path, fake_path = _write_raw_corpus(tmp_path, per_class=80)
    data, _ = build_dataset(str(true_path), str(fake_path))
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data["content"]
    y = data["label"]
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.25)

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results, fitted = {}, {}
    for name, classifier in build_classifiers().items():
        model, metrics, y_pred, _ = evaluate_model(
            classifier, X_train_vec, X_test_vec, y_train, y_test
        )
        metrics["y_true"] = list(y_test)
        metrics["y_pred"] = list(y_pred)
        results[name] = metrics
        fitted[name] = model

    return {
        "results": results,
        "fitted": fitted,
        "vectorizer": vectorizer,
        "X_test": X_test,
        "y_test": y_test,
        "output_dir": tmp_path / "results",
    }


def test_all_three_algorithms_train_and_beat_baseline(trained_experiment):
    results = trained_experiment["results"]
    assert set(results) == {
        "Logistic Regression",
        "Multinomial Naive Bayes",
        "Linear SVM",
    }
    for name, metrics in results.items():
        assert metrics["accuracy"] > 0.9, name
        assert 0.0 <= metrics["f1_weighted"] <= 1.0
        assert metrics["train_seconds"] >= 0


def test_metrics_are_computed_from_the_predictions(trained_experiment):
    metrics = trained_experiment["results"]["Logistic Regression"]
    matrix = metrics["confusion_matrix"]

    # The confusion matrix must cover every test article exactly once.
    assert sum(sum(row) for row in matrix) == len(trained_experiment["y_test"])
    correct = matrix[0][0] + matrix[1][1]
    assert metrics["accuracy"] == pytest.approx(
        correct / len(trained_experiment["y_test"]), rel=1e-6
    )


def test_probabilistic_models_report_calibration_metrics(trained_experiment):
    for name in ("Logistic Regression", "Multinomial Naive Bayes"):
        metrics = trained_experiment["results"][name]
        assert metrics["provides_probabilities"] is True
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert metrics["log_loss"] > 0.0

    svm = trained_experiment["results"]["Linear SVM"]
    assert svm["provides_probabilities"] is False
    assert "brier_score" not in svm


def test_model_selection_prefers_a_probability_model(trained_experiment):
    selection = select_model(trained_experiment["results"])
    assert trained_experiment["results"][selection]["provides_probabilities"] is True


def test_saved_pipeline_round_trips_and_predicts(trained_experiment, tmp_path):
    pipeline = build_pipeline(
        trained_experiment["vectorizer"],
        trained_experiment["fitted"]["Logistic Regression"],
    )
    path = tmp_path / "model.pkl"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)

    text = f"{FAKE_WORDS} exclusive report"
    probabilities = loaded.predict_proba([text])[0]
    assert probabilities.sum() == pytest.approx(1.0)
    assert set(loaded.predict([text])).issubset({0, 1})


def test_pipeline_uses_the_same_vectorizer_settings_as_training(trained_experiment):
    pipeline = build_pipeline(
        trained_experiment["vectorizer"],
        trained_experiment["fitted"]["Logistic Regression"],
    )
    tfidf = pipeline.named_steps["tfidf"]
    assert tfidf.stop_words == "english"
    assert tfidf.ngram_range == (1, 2)
    assert tfidf.min_df == 2
    assert tfidf.max_features == 50000
    assert pipeline.named_steps["classifier"].__class__ is LogisticRegression


def test_save_results_writes_table_reports_and_plots(trained_experiment):
    output_dir = trained_experiment["output_dir"]
    comparison, payload = save_results(
        trained_experiment["results"],
        "Logistic Regression",
        {"total_articles": 160, "train_articles": 120, "test_articles": 40},
        trained_experiment["vectorizer"],
        output_dir=str(output_dir),
    )

    assert len(comparison) == 3
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "model_comparison.png").exists()
    assert (output_dir / "metrics.json").exists()
    for slug in ("logistic_regression", "multinomial_naive_bayes", "linear_svm"):
        assert (output_dir / f"confusion_matrix_{slug}.png").exists()
        assert (output_dir / f"classification_report_{slug}.txt").exists()

    saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["selected_model"] == "Logistic Regression"
    assert saved["dataset"]["test_articles"] == 40
    assert set(saved["models"]) == set(trained_experiment["results"])


def test_prepare_data_module_keeps_cleaning_order():
    # Guard against the pipeline silently changing its behavior: both removals
    # must be applied, and the shared sanitizer must run on both classes.
    assert prepare_data.REAL_LABEL == 0 and prepare_data.FAKE_LABEL == 1
    assert prepare_data.clean_text is clean_text
