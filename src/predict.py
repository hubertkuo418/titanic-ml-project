import pandas as pd
import joblib
import os

# Get project root directory (two levels up from current file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to trained model and dataset
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
data_path = os.path.join(BASE_DIR, "data", "train.csv")

# Load trained ML model
model = joblib.load(model_path)

# Load dataset
df = pd.read_csv(data_path)

# Remove target column if it exists (for inference consistency)
if "Survived" in df.columns:
    df = df.drop(columns=["Survived"])

# Drop irrelevant or high-cardinality features
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handle missing values in numerical and categorical columns
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Feature engineering: create family-related features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Convert categorical variables into numerical format (one-hot encoding)
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Make predictions using trained model
pred = model.predict(df)

# Attach predictions back to dataframe
df["Prediction"] = pred

# Show first few predictions
print(df[["Prediction"]].head())