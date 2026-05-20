import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Get project root directory (two levels up from current file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to dataset
data_path = os.path.join(BASE_DIR, "data", "train.csv")

# Load dataset
df = pd.read_csv(data_path)

# Drop unnecessary or high-cardinality features
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Feature engineering: create additional meaningful features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Convert categorical variables into numerical format
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Split features and target variable
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save trained model to disk
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
joblib.dump(model, model_path)

# Print confirmation message
print("Model saved:", model_path)