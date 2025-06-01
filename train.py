# train.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
df = pd.read_csv("data/iris.csv")

# Feature-target split
X = df.drop(columns=["species"])
y = df["species"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print("Model trained and saved as models/model.joblib")
