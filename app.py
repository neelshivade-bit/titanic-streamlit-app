
import streamlit as st
import pickle
import numpy as np

# Load model
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode inputs
sex_encoded = 0 if sex == "male" else 1
embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]

# Predict
if st.button("Predict"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
    prediction = model.predict(input_data)[0]
    result = "Survived 🎉" if prediction == 1 else "Did Not Survive 💀"
    st.subheader(f"Prediction: {result}")
