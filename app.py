import streamlit as st
import pandas as pd
import joblib

# Load trained model from disk
model = joblib.load("models/model.pkl")

# App title
st.title("Titanic Survival Prediction App")
st.write("Enter passenger information to predict survival.")

# =========================
# User Input Section
# =========================

# Passenger class
pclass = st.selectbox("Pclass", [1, 2, 3])

# Gender
sex = st.selectbox("Sex", ["male", "female"])

# Age input (slider for UI convenience)
age = st.slider("Age", 0, 80, 25)

# Number of siblings/spouses aboard
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)

# Number of parents/children aboard
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)

# Ticket fare
fare = st.slider("Fare", 0, 500, 50)

# Port of embarkation
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# =========================
# Feature Engineering
# =========================

# Family size = siblings/spouses + parents/children + self
family_size = sibsp + parch + 1

# Binary feature: whether passenger is alone
is_alone = 1 if family_size == 1 else 0

# =========================
# Model Input Construction
# =========================

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "Sex_male": [1 if sex == "male" else 0],
    "Embarked_Q": [1 if embarked == "Q" else 0],
    "Embarked_S": [1 if embarked == "S" else 0],
})

# =========================
# Prediction Section
# =========================

if st.button("Predict"):
    # Predict survival class (0 = not survived, 1 = survived)
    prediction = model.predict(input_data)[0]

    # Predict probability of survival
    prob = model.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 1:
        st.success(f"Survived! (prob = {prob:.2f})")
    else:
        st.error(f"Did not survive (prob = {prob:.2f})")