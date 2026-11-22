import pandas as pd

# Load the REAL news dataset
real_news = pd.read_csv("data/raw/True.csv")

# Load the FAKE news dataset
fake_news = pd.read_csv("data/raw/Fake.csv")

# Display basic information
print("REAL NEWS DATA")
print(real_news.head())

print("\nFAKE NEWS DATA")
print(fake_news.head())

# Display dataset sizes
print("\nNumber of real articles:", len(real_news))
print("Number of fake articles:", len(fake_news))

# Display column names
print("\nColumns:", real_news.columns.tolist())