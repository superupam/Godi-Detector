# ==========================================
# GODI DETECTOR
# Data Preparation & Sanitization Pipeline
# ==========================================

import os
import re
import pandas as pd

# STEP 1: Load both raw datasets
print("Step 1: Loading raw datasets...")
real_news = pd.read_csv("data/raw/True.csv")
fake_news = pd.read_csv("data/raw/Fake.csv")

print(f"Loaded Real articles: {len(real_news):,}")
print(f"Loaded Fake articles: {len(fake_news):,}")

# STEP 2: Assign target labels
# 0 = Real News, 1 = Fake News
real_news["label"] = 0
fake_news["label"] = 1

# STEP 3: Clean & Sanitize Text (Prevent Data Leakage)
print("\nStep 2: Sanitizing text to remove publisher leakage and artifacts...")

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
AGENCY_CREDIT_RE = re.compile(r"(?i)(?:(?<![A-Za-z])[A-Z][\w'’’.\-]*\s*/\s*)+getty images\.?")


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


for _frame in (real_news, fake_news):
    _frame["title"] = _frame["title"].fillna("").apply(clean_text)
    _frame["text"] = _frame["text"].fillna("").apply(clean_text)
print("Sanitization completed!")

# STEP 4: Combine title and text into a unified content column
print("\nStep 3: Combining title and text...")

real_news["content"] = (real_news["title"] + " " + real_news["text"]).str.strip()
fake_news["content"] = (fake_news["title"] + " " + fake_news["text"]).str.strip()

# Combine both datasets
news_data = pd.concat([real_news, fake_news], ignore_index=True)
initial_count = len(news_data)
print(f"Total combined articles before deduplication: {initial_count:,}")

# STEP 5: Remove duplicates and empty articles
print("\nStep 4: Checking and removing duplicates...")
duplicate_count = news_data["content"].duplicated().sum()
print(f"Duplicate articles found: {duplicate_count:,}")

news_data = news_data.drop_duplicates(subset=["content"]).reset_index(drop=True)
# Remove empty or trivially short articles (< 20 characters)
news_data = news_data[news_data["content"].str.len() > 20].reset_index(drop=True)

final_count = len(news_data)
print(f"Articles retained after cleaning: {final_count:,} (Removed {initial_count - final_count:,})")

# STEP 6: Shuffle the dataset
print("\nStep 5: Shuffling dataset...")
news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)

# STEP 7: Summary and distribution
print("\nCleaned Class Distribution:")
distribution = news_data["label"].value_counts()
print(f"  Real News (0): {distribution.get(0, 0):,} ({distribution.get(0, 0) / final_count * 100:.1f}%)")
print(f"  Fake News (1): {distribution.get(1, 0):,} ({distribution.get(1, 0) / final_count * 100:.1f}%)")

# STEP 8: Save the cleaned dataset
print("\nStep 6: Saving cleaned dataset...")
os.makedirs("data/processed", exist_ok=True)
news_data.to_csv("data/processed/news_dataset.csv", index=False)
print("Saved to data/processed/news_dataset.csv successfully!")
print("==========================================")
print("DATASET CLEANING & PREPARATION COMPLETE!")
print("==========================================")
