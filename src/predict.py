import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
data_path = os.path.join(BASE_DIR, "data", "train.csv")

# 載入 model
model = joblib.load(model_path)

df = pd.read_csv(data_path)

if "Survived" in df.columns:
    df = df.drop(columns=["Survived"])

df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

pred = model.predict(df)

df["Prediction"] = pred

print(df[["Prediction"]].head())