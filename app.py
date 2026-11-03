import streamlit as st
import joblib

# Load trained model
model = joblib.load("iris_model.pkl")

# Application title
st.title("Iris Flower Prediction")

# Get flower measurements
sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

# Prediction button
if st.button("Predict"):

    pred = model.predict([[sl, sw, pl, pw]])

    flowers = ["Setosa", "Versicolor", "Virginica"]

    st.success(flowers[pred[0]])