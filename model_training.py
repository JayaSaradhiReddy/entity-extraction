import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
data = pd.read_csv("itsm_tickets.csv")

# Combine summary and description
data["text"] = data["Summary"] + " " + data["Description"]

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Function to train and save a model
def train_model(target_column, filename):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    model = Pipeline([("vectorizer", vectorizer), ("classifier", LogisticRegression())])
    
    model.fit(data["text"], data[target_column])  # Train the model
    joblib.dump(model, f"models/{filename}")  # Save the model

# Train models for Context, Server, and Software
train_model("Context", "context_model.pkl")
train_model("Server", "server_model.pkl")
#train_model("Software", "software_model.pkl")

print("Training completed. Models saved in 'models' directory.")
