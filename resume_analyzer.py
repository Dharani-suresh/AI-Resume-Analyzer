import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Step 1: Load JSONL dataset
# -------------------------------
resumes = []
with open("resumes_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        resumes.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(resumes)

# Keep only 'Text' and 'Category'
df = df[["Text", "Category"]]

# Rename for convenience
df = df.rename(columns={"Text": "Resume"})

print("Dataset loaded successfully!")
print(df.head())

# -------------------------------
# Step 2: Prepare features
# -------------------------------
X = df["Resume"]
y = df["Category"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# -------------------------------
# Step 3: Train Naive Bayes Model
# -------------------------------
model = MultinomialNB()
model.fit(X_vectorized, y)

print("Model trained successfully!")

# -------------------------------
# Step 4: Make Sample Prediction
# -------------------------------
sample_resume = ["Python machine learning data analysis SQL"]
prediction = model.predict(vectorizer.transform(sample_resume))

print("Predicted Job Category:", prediction[0])