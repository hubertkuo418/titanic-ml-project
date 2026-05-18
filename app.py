import streamlit as st
import pandas as pd
import joblib

# 載入模型
model = joblib.load("models/model.pkl")

st.title("Titanic Survival Prediction App")
st.write("Enter passenger information to predict survival.")

# --- 使用者輸入 ---
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Fare", 0, 500, 50)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# --- feature engineering ---
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# --- dataframe ---
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

# --- 預測 ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f" Survived! (prob = {prob:.2f})")
    else:
        st.error(f" Did not survive (prob = {prob:.2f})")