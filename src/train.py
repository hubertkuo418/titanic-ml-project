import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 🔥 取得專案根目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🔥 正確讀資料路徑
data_path = os.path.join(BASE_DIR, "data", "train.csv")

df = pd.read_csv(data_path)

df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 存模型
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
joblib.dump(model, model_path)

print("Model saved:", model_path)