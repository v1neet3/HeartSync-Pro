import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the pre-trained model
model = keras.models.load_model('heart_disease_model.hdf5')
df = pd.read_csv("Heart_Disease_Prediction.csv")

x = df.iloc[:, :-1]  # Select all columns except the last one (features)
y = df.iloc[:, -1]  # Select the last column (target variable)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.35)

# Function to preprocess user input and make predictions
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    sc_x = StandardScaler()
    sc_x.fit(x_train)
    user_input_scaled = sc_x.transform(input_data)
    prediction = model.predict(user_input_scaled)
    return prediction

# Create a Streamlit app
st.title("Heart Disease Predictor")

# Input fields for user
age = st.slider("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex(male=0, femlae=1)", [0, 1])
cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", min_value=90, max_value=200, value=120)
chol = st.slider("Cholesterol (Chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar (FBS)", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (restECG)", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", min_value=70, max_value=420, value=150)
exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1])
oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.2, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.slider("Number of Major Vessels Colored by Fluoroscopy (Ca)", min_value= 0, max_value= 3, value=0)
thal = st.selectbox("Thallium Stress Test (thal)", [0, 1, 2, 3, 4, 5, 6, 7])

if st.button("Predict"):
    prediction = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if prediction >= 0.5:
        st.write("Based on the input data, it is likely that you have heart disease.")
    else:
        st.write("Based on the input data, it is likely that you do not have heart disease.")
