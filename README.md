# Titanic ML Project

> End-to-end machine learning pipeline for predicting passenger survival on the Titanic dataset.

---

## 🚀 Overview

This project implements a complete ML workflow:

- Data preprocessing
- Feature engineering
- Model training & comparison
- Evaluation
- Streamlit deployment

Goal: build a **production-style ML pipeline**, not just a training script.

---

## 🎯 Problem Type

Binary Classification  
Target: `Survived (0 = No, 1 = Yes)`

---

## 🧠 ML Models

- Logistic Regression
- LDA
- KNN
- Decision Tree
- Bagging
- Random Forest
- SVM

---

## ⚙️ Pipeline Design

```
Data Loading
   ↓
Missing Value Handling
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Prediction
```

Key idea: **structured ML pipeline from raw data to deployment**

---

## 📊 Key Results

Best Model: Random Forest / SVM

### ROC Curve
![roc](assets/ROC_Curve_Comparison.png)

### Confusion Matrix
![cm](assets/Confusion_Matrix_Random_Forest.png)

### Feature Importance
![fi](assets/Feature_Importance_Random_Forest.png)

---

## 🎯 Feature Importance

Most important features:
- Sex
- Pclass
- Fare
- Age

---

## 🖥 Streamlit App

### Demo
![demo](assets/streamlit_demo.png)

### Run locally
```bash
streamlit run app.py
```

### Features
- Interactive input form
- Real-time prediction
- Survival probability output

---

## ▶️ How to Run

```bash
pip install -r requirements.txt

# Train model
python src/train.py

# Run inference
python src/predict.py

# Launch UI
streamlit run app.py
```

---

## 🧰 Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit

---

## 🧩 Key Concepts

- End-to-end ML pipeline design
- Feature engineering impact
- Model comparison strategy
- Overfitting awareness
- Basic model deployment

---

## 🔮 Future Work

- Hyperparameter tuning (GridSearchCV)
- Cross-validation
- Feature selection optimization
- Model stacking

---

## 👤 Author

Built by: Hubert Kuo  
Focus: Computer Vision / AI Systems / Machine Learning