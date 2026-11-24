# ==========================================
# GODI DETECTOR
# Exploratory Data Analysis
# ==========================================
#
# Inspects the raw Kaggle files and the prepared dataset, prints the numbers
# used in the README, and writes the EDA figures to results/.
#
# Run from the project root:  python explore_data.py

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display needed
import matplotlib.pyplot as plt
import pandas as pd

RAW_REAL_PATH = "data/raw/True.csv"
RAW_FAKE_PATH = "data/raw/Fake.csv"
PROCESSED_PATH = "data/processed/news_dataset.csv"
RESULTS_DIR = "results"
SUMMARY_PATH = "results/eda_summary.json"

LABEL_NAMES = {0: "Real", 1: "Fake"}


def _word_stats(series):
    """Describe article length in words, ignoring empty strings."""
    lengths = series.fillna("").astype(str).str.split().str.len()
    return {
        "mean": round(float(lengths.mean()), 2),
        "median": round(float(lengths.median()), 2),
        "p90": round(float(lengths.quantile(0.90)), 2),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
    }


def _plot_label_distribution(raw_counts, processed_counts, path):
    """Two panels: corpus composition before and after cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, counts, title in (
        (axes[0], raw_counts, "Raw Kaggle files"),
        (axes[1], processed_counts, "After cleaning"),
    ):
        labels = ["Real (0)", "Fake (1)"]
        values = [counts.get(0, 0), counts.get(1, 0)]
        bars = ax.bar(labels, values, color=["#10B981", "#EF4444"])
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_ylabel("Articles")
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Label distribution — balanced corpus, no resampling needed")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_lengths(raw_frames, processed, path):
    """Word-count distributions per class, raw vs cleaned."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = range(0, 1601, 50)

    for label, frame in raw_frames.items():
        words = frame["text"].fillna("").astype(str).str.split().str.len()
        axes[0].hist(words, bins=bins, alpha=0.55, label=LABEL_NAMES[label])
    axes[0].set_title("Raw article bodies (words)")
    axes[0].set_xlabel("Words per article")
    axes[0].set_ylabel("Articles")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for label, group in processed.groupby("label"):
        words = group["content"].astype(str).str.split().str.len()
        axes[1].hist(words, bins=bins, alpha=0.55, label=LABEL_NAMES[label])
    axes[1].set_title("Cleaned title + body (words)")
    axes[1].set_xlabel("Words per article")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_subjects(raw_frames, path):
    """Where the raw articles came from: Kaggle subject categories."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (label, frame) in zip(axes, raw_frames.items()):
        counts = frame["subject"].value_counts()
        bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#3B82F6")
        ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
        ax.set_title(f"{LABEL_NAMES[label]} — raw subjects")
        ax.set_xlabel("Articles")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    print("Loading raw datasets...")
    real_news = pd.read_csv(RAW_REAL_PATH)
    fake_news = pd.read_csv(RAW_FAKE_PATH)
    raw_frames = {0: real_news, 1: fake_news}

    print(f"Real (True.csv): {len(real_news):,} rows, columns {list(real_news.columns)}")
    print(f"Fake (Fake.csv): {len(fake_news):,} rows, columns {list(fake_news.columns)}")
    print(f"Combined: {len(real_news) + len(fake_news):,} articles")

    print("\nMissing values (raw):")
    print(
        pd.DataFrame(
            {
                name: frame.isna().sum()
                for name, frame in (("True.csv", real_news), ("Fake.csv", fake_news))
            }
        ).to_string()
    )

    print("\nExact duplicate rows (raw):")
    for name, frame in (("True.csv", real_news), ("Fake.csv", fake_news)):
        print(f"  {name}: {frame.duplicated().sum():,}")

    print("\nRaw subject distribution:")
    for label, frame in raw_frames.items():
        print(f"  {LABEL_NAMES[label]}: {frame['subject'].value_counts().to_dict()}")

    print("\nRaw body length in words:")
    for label, frame in raw_frames.items():
        print(f"  {LABEL_NAMES[label]}: {_word_stats(frame['text'])}")

    processed = pd.read_csv(PROCESSED_PATH)
    print(f"\nPrepared dataset: {len(processed):,} rows, columns {list(processed.columns)}")
    processed_counts = processed["label"].value_counts().sort_index().to_dict()
    print(f"Class distribution: {processed_counts}")
    print(f"Exact duplicate contents remaining: {processed['content'].duplicated().sum():,}")
    print("Cleaned content length in words:")
    for label, group in processed.groupby("label"):
        print(f"  {LABEL_NAMES[label]}: {_word_stats(group['content'])}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _plot_label_distribution(
        {label: len(frame) for label, frame in raw_frames.items()},
        processed_counts,
        os.path.join(RESULTS_DIR, "eda_label_distribution.png"),
    )
    _plot_lengths(raw_frames, processed, os.path.join(RESULTS_DIR, "eda_article_lengths.png"))
    _plot_subjects(raw_frames, os.path.join(RESULTS_DIR, "eda_subjects.png"))

    summary = {
        "raw_rows": {
            name: int(len(frame))
            for name, frame in (("True.csv", real_news), ("Fake.csv", fake_news))
        },
        "raw_total": int(len(real_news) + len(fake_news)),
        "raw_missing_values": {
            "True.csv": {k: int(v) for k, v in real_news.isna().sum().items()},
            "Fake.csv": {k: int(v) for k, v in fake_news.isna().sum().items()},
        },
        "raw_duplicate_rows": {
            "True.csv": int(real_news.duplicated().sum()),
            "Fake.csv": int(fake_news.duplicated().sum()),
        },
        "raw_subjects": {
            name: {k: int(v) for k, v in frame["subject"].value_counts().items()}
            for name, frame in (("True.csv", real_news), ("Fake.csv", fake_news))
        },
        "raw_body_words": {
            LABEL_NAMES[label]: _word_stats(frame["text"])
            for label, frame in raw_frames.items()
        },
        "processed_rows": int(len(processed)),
        "processed_class_counts": {str(k): int(v) for k, v in processed_counts.items()},
        "processed_duplicate_contents": int(processed["content"].duplicated().sum()),
        "processed_content_words": {
            LABEL_NAMES[label]: _word_stats(group["content"])
            for label, group in processed.groupby("label")
        },
        "figures": [
            "results/eda_label_distribution.png",
            "results/eda_article_lengths.png",
            "results/eda_subjects.png",
        ],
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEDA figures and summary written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
