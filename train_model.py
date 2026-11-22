# ==========================================
# GODI DETECTOR
# First Machine Learning Classifier
# ==========================================

# BLOCK 1: Import libraries

import pandas as pd
import joblib
import os
import json
import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# BLOCK 2: Load the prepared dataset

print("Loading dataset...")

news_data = pd.read_csv(
    "data/processed/news_dataset.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", news_data.shape)


# BLOCK 3: Prepare the text

print("\nPreparing text data...")

# Handle missing values
news_data["title"] = news_data["title"].fillna("")
news_data["text"] = news_data["text"].fillna("")

# Combine title and article text
news_data["content"] = (
    news_data["title"] + " " + news_data["text"]
)

# Remove empty articles
news_data = news_data[
    news_data["content"].str.strip() != ""
]

# Separate input and output

X = news_data["content"]
y = news_data["label"]

print("Text preparation completed!")
print("Total articles:", len(X))

# Check duplicate articles

print("\nChecking duplicate articles...")

duplicates = news_data["content"].duplicated().sum()

print("Duplicate articles:", duplicates)


# BLOCK 4: Split the dataset

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))


# BLOCK 5: Build and train the ML model

print("\nBuilding ML model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

print("Training started...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# BLOCK 6: Evaluate the model

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(f"{accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Real News", "Fake News"]
    )
)

print("\nCONFUSION MATRIX:")

cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print(cm)
print("\nINTERPRETATION:")

print("Correct Real News:",
      cm[0][0])

print("Real News classified as Fake:",
      cm[0][1])

print("Fake News classified as Real:",
      cm[1][0])

print("Correct Fake News:",
      cm[1][1])


# BLOCK 7: Save the trained model (versioned + latest)

print("\nSaving trained model...")

os.makedirs("models", exist_ok=True)

# --- Versioned snapshot (never overwritten) ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_path = f"models/godi_news_classifier_{timestamp}.pkl"
joblib.dump(model, versioned_path)
print(f"Versioned model saved to: {versioned_path}")

# --- Always-latest alias ---
latest_path = "models/godi_news_classifier.pkl"
joblib.dump(model, latest_path)
print(f"Latest model saved to:    {latest_path}")

# --- Save metadata for the API status endpoint ---
meta = {
    "trained_at": timestamp,
    "accuracy": round(float(accuracy * 100), 4),
    "dataset_size": int(len(X)),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "versioned_file": versioned_path,
}
with open("models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"Model metadata saved to: models/model_meta.json")

print("\n================================")
print("GODI DETECTOR MODEL SAVED!")
print("================================")