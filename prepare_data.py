# ==========================================
# GODI DETECTOR
# Data Preparation & Sanitization Pipeline
# ==========================================
#
# data/raw/True.csv + data/raw/Fake.csv
#   -> label leakage removal
#   -> title + text combined
#   -> duplicates and stubs dropped
#   -> data/processed/news_dataset.csv + results/dataset_stats.json
#
# Run from the project root:  python prepare_data.py

import json
import os
import re

import pandas as pd

RAW_REAL_PATH = "data/raw/True.csv"
RAW_FAKE_PATH = "data/raw/Fake.csv"
PROCESSED_PATH = "data/processed/news_dataset.csv"
STATS_PATH = "results/dataset_stats.json"

# Labels: 0 = Real News, 1 = Fake News
REAL_LABEL = 0
FAKE_LABEL = 1

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TWITTER_MEDIA_RE = re.compile(r"pic\.twitter\.com/\S+", re.IGNORECASE)

# Labels are matched without a leading \b: the HTML-stripping leaves credits glued
# to the previous sentence, e.g. "...June 8, 2017Featured image via ...".
CREDIT_LABEL_RE = re.compile(r"(?i)featured\s+(?:image|photo)")
PHOTO_CREDIT_RE = re.compile(
    r"(?i)(?:featured\s+)?(?:photos?(?:graph)?|image)(?:\s+credit)?\s*:?\s*"
    r"(?:(?:by|of|via)\s+)?[^!?]{0,120}?"
    r"(?:getty images|flickr|wikimedia commons|associated press|afp)\.?"
)
AGENCY_CREDIT_RE = re.compile(r"(?i)(?:(?<![A-Za-z])[A-Z][\w'’’. \-]*\s*/\s*)+getty images\.?")


def strip_media_credits(text):
    """
    Remove image/photo credits that reveal a fake article's source CMS.
    Example: 'Featured image: YouTube', 'Photo by Jane Doe/Getty Images'.
    """
    if not isinstance(text, str):
        return ""

    label = CREDIT_LABEL_RE.search(text)
    if label and len(text) - label.start() <= 240:
        # Credit block that runs (almost) to the end of the article -> drop the tail.
        text = text[: label.start()]
    else:
        # Short inline credit ('Featured image is a screenshot ...') -> drop the label only.
        text = re.sub(
            r"(?i)featured\s+(?:image|photo)(?:\s+(?:is|was|from|via|by|of|:))?"
            r"(?:\s+(?:a|an|the))?"
            r"(?:\s+(?:screenshot|photo(?:graph)?|picture|image|video|screen|youtube|flickr|twitter))?",
            " ",
            text,
        )

    # Photo / agency credits anywhere in the text
    text = PHOTO_CREDIT_RE.sub(" ", text)
    text = AGENCY_CREDIT_RE.sub(" ", text)
    text = re.sub(r"(?i)images?\s+merged\s+by\s+[^.]{0,80}\.?", " ", text)
    return text


def strip_publisher_mentions(text):
    """
    Remove Reuters datelines and wire tags that leak the label of real articles.
    Example: 'WASHINGTON (Reuters) - The president...' -> 'The president...'
    """
    if not isinstance(text, str):
        return ""
    # Leading datelines like "CITY (Reuters) - "
    text = re.sub(r"(?i)^[A-Z\s,./'-]{2,}?\(reuters\)\s*[-:]\s*", "", text)
    text = re.sub(r"(?i)^\s*\(reuters\)\s*[-:]\s*", "", text)
    # 'Reuters/Ipsos poll' -> 'Ipsos poll', 'Reuters/ANES' -> 'ANES'
    text = re.sub(r"(?i)(?<![A-Za-z])reuters\s*/\s*ipsos(?![A-Za-z])", "Ipsos", text)
    text = re.sub(r"(?i)(?<![A-Za-z])reuters\s*/\s*", "", text)
    # Any remaining mention, including the possessive "Reuters'"
    text = re.sub(r"(?i)(?<![A-Za-z])reuters'?s?(?![A-Za-z])", "", text)
    # Tidy the punctuation/whitespace left behind by the removals
    text = re.sub(r"(?i)\b(for|with|by|to|of|and|from)\s+-\s+", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([:;,.!?])", r"\1", text)
    return text


def clean_text(text):
    """Single sanitizer applied to BOTH classes so no label-specific artifact survives."""
    if not isinstance(text, str):
        return ""
    text = strip_media_credits(text)
    text = TWITTER_MEDIA_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = strip_publisher_mentions(text)
    return re.sub(r"\s{2,}", " ", text).strip()


def load_raw_news(true_path=RAW_REAL_PATH, fake_path=RAW_FAKE_PATH):
    """Load both raw files and assign the target labels."""
    real_news = pd.read_csv(true_path)
    fake_news = pd.read_csv(fake_path)
    real_news["label"] = REAL_LABEL
    fake_news["label"] = FAKE_LABEL
    return real_news, fake_news


def build_dataset(true_path=RAW_REAL_PATH, fake_path=RAW_FAKE_PATH):
    """
    Run the full cleaning pipeline.

    Returns:
        (cleaned DataFrame, stats dict with the real row counts at each stage)
    """
    real_news, fake_news = load_raw_news(true_path, fake_path)
    stats = {
        "raw_real": int(len(real_news)),
        "raw_fake": int(len(fake_news)),
        "raw_total": int(len(real_news) + len(fake_news)),
        "raw_missing_titles": int(
            real_news["title"].isna().sum() + fake_news["title"].isna().sum()
        ),
        "raw_missing_texts": int(
            real_news["text"].isna().sum() + fake_news["text"].isna().sum()
        ),
    }

    # Sanitize text (prevents publisher/credit leakage) for both classes.
    for frame in (real_news, fake_news):
        frame["title"] = frame["title"].fillna("").apply(clean_text)
        frame["text"] = frame["text"].fillna("").apply(clean_text)

    # Combine title and text into a unified content column.
    real_news["content"] = (real_news["title"] + " " + real_news["text"]).str.strip()
    fake_news["content"] = (fake_news["title"] + " " + fake_news["text"]).str.strip()

    news_data = pd.concat([real_news, fake_news], ignore_index=True)
    stats["combined_total"] = int(len(news_data))

    # Duplicates and empty/trivially short articles.
    stats["duplicates_removed"] = int(news_data["content"].duplicated().sum())
    news_data = news_data.drop_duplicates(subset=["content"]).reset_index(drop=True)

    too_short = news_data["content"].str.len() <= 20
    stats["short_removed"] = int(too_short.sum())
    news_data = news_data[~too_short].reset_index(drop=True)

    stats["cleaned_total"] = int(len(news_data))
    stats["removed_total"] = int(stats["combined_total"] - stats["cleaned_total"])

    # Shuffle with a fixed seed so the split is reproducible.
    news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)

    stats["real_retained"] = int((news_data["label"] == REAL_LABEL).sum())
    stats["fake_retained"] = int((news_data["label"] == FAKE_LABEL).sum())
    stats["real_pct"] = round(stats["real_retained"] / stats["cleaned_total"] * 100, 2)
    stats["fake_pct"] = round(stats["fake_retained"] / stats["cleaned_total"] * 100, 2)

    return news_data, stats


def main():
    print("Step 1: Loading raw datasets...")
    news_data, stats = build_dataset()

    print(f"Loaded Real articles: {stats['raw_real']:,}")
    print(f"Loaded Fake articles: {stats['raw_fake']:,}")
    print(f"Total combined articles before deduplication: {stats['raw_total']:,}")
    print(f"Duplicate articles found: {stats['duplicates_removed']:,}")
    print(
        f"Articles retained after cleaning: {stats['cleaned_total']:,} "
        f"(Removed {stats['removed_total']:,})"
    )

    print("\nCleaned Class Distribution:")
    print(f"  Real News (0): {stats['real_retained']:,} ({stats['real_pct']:.1f}%)")
    print(f"  Fake News (1): {stats['fake_retained']:,} ({stats['fake_pct']:.1f}%)")

    print("\nSaving cleaned dataset...")
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    news_data.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved to {PROCESSED_PATH} successfully!")

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset statistics saved to: {STATS_PATH}")

    print("==========================================")
    print("DATASET CLEANING & PREPARATION COMPLETE!")
    print("==========================================")


if __name__ == "__main__":
    main()
