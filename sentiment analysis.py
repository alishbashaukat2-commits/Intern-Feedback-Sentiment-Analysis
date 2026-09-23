import pandas as pd

# Dataset path
file_path = r"E:\Desktop\remote internship\task 2\intern_feedback_sentiment_dataset (1).csv"

# Load dataset
df = pd.read_csv(file_path)

# Basic information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

print("\nSentiment Percentage:")
print(df["Sentiment"].value_counts(normalize=True) * 100)
#EDA
print(df["Sentiment"].value_counts())

print(df["Category"].value_counts())

print(df["Department"].value_counts())

print(df["Internship_Duration_Months"].value_counts().sort_index())

print(df["Overall_Rating"].value_counts().sort_index())

print(pd.crosstab(df["Category"], df["Sentiment"]))

print(pd.crosstab(df["Department"], df["Sentiment"]))
#preparing the text
df["Feedback_Text"] = df["Feedback_Text"].str.lower()

df["Feedback_Text"] = df["Feedback_Text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)

df["Feedback_Text"] = df["Feedback_Text"].str.replace(r"\s+", " ", regex=True)

df["Feedback_Text"] = df["Feedback_Text"].str.strip()

print(df["Feedback_Text"].head())
#spliting dataset
from sklearn.model_selection import train_test_split

X = df["Feedback_Text"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

print("\nTraining sentiment:")
print(y_train.value_counts())

print("\nTesting sentiment:")
print(y_test.value_counts())
#TF-IDF converting our feedback text into numbers that the Logistic Regression model can understand.
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Fit on training data and transform
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data
X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

#starting training our model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

import transformers

print("Transformers version:", transformers.__version__)
import transformers
import torch

print("Transformers version:", transformers.__version__)
print("PyTorch version:", torch.__version__)
from transformers import pipeline

# Load pretrained sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Test one feedback
text = "I really enjoyed my internship and learned a lot."

result = sentiment_model(text)

print(result)
from transformers import pipeline

# Load 3-class sentiment model
sentiment_model_3class = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Test one feedback
text = "I really enjoyed my internship and learned a lot."

result = sentiment_model_3class(text)

print(result)

# Step 11 - Transformer Predictions
transformer_predictions = []

for text in X_test:
    result = sentiment_model_3class(text)[0]
    transformer_predictions.append(result["label"].capitalize())

print("Transformer predictions completed!")
print("Number of predictions:", len(transformer_predictions))

# Step 12 - Transformer Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

transformer_accuracy = accuracy_score(y_test, transformer_predictions)

print("Transformer Accuracy:", transformer_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, transformer_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, transformer_predictions))
# Step 13: Model Comparison

import pandas as pd
import matplotlib.pyplot as plt

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Pretrained Transformer"],
    "Accuracy": [0.99, 0.77],
    "Macro F1": [0.99, 0.76]
})

print("\nModel Comparison:")
print(comparison)

# Bar chart
comparison.set_index("Model")[["Accuracy", "Macro F1"]].plot(kind="bar")

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 14: Analyze Negative and Neutral Feedback

print("\nNegative Feedback Examples:")
negative_feedback = df[df["Sentiment"] == "Negative"]["Feedback_Text"]

for feedback in negative_feedback.head(10):
    print("-", feedback)

print("\nNeutral Feedback Examples:")
neutral_feedback = df[df["Sentiment"] == "Neutral"]["Feedback_Text"]

for feedback in neutral_feedback.head(10):
    print("-", feedback)

# Sentiment Distribution

sentiment_counts = df["Sentiment"].value_counts()

print("\nSentiment Distribution:")
print(sentiment_counts)

print("\nSentiment Percentages:")
print((sentiment_counts / len(df) * 100).round(2))    
