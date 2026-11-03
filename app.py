import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Iris Flower Prediction")

# Application title
st.title("🌸 Iris Flower Prediction")
st.write("Enter the flower measurements to predict its species.")

# Load trained model
try:
    model = joblib.load("iris_model.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Get flower measurements
sl = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sw = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
pl = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
pw = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

# Prediction button
if st.button("Predict"):

    data = np.array([[sl, sw, pl, pw]])

    try:
        pred = model.predict(data)

        flowers = ["Setosa", "Versicolor", "Virginica"]

        st.success(f"Predicted Flower: {flowers[int(pred[0])]}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
