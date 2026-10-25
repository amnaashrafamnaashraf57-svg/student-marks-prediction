
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Student Marks Prediction")
st.write("Predict student marks based on study hours.")

# Sample training data
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [15, 25, 35, 45, 55, 65, 75, 85]
}

df = pd.DataFrame(data)

# Prepare input and output
X = df[["Hours"]]
y = df["Marks"]

# Train the machine learning model
model = LinearRegression()
model.fit(X, y)

# Get study hours from the user
hours = st.number_input(
    "Enter study hours:",
    min_value=0.0,
    max_value=24.0,
    value=6.5,
    step=0.5
)

# Predict marks
new_data = pd.DataFrame({"Hours": [hours]})
prediction = model.predict(new_data)

st.success(f"Predicted Marks: {prediction[0]:.2f}")
