import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the trained Random Forest model
@st.cache_resource
def load_model():
    with open("RF.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# 2. Set up the App Title and Description
st.title("🍇 Raisin Class Prediction App")
st.write("""
This application uses a Random Forest Machine Learning model to classify raisins into two categories: 
**Besni** or **Kecimen**, based on their geometrical features.
""")

st.subheader("Enter Raisin Features:")

# 3. Create input elements for all 7 features used during training
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", min_value=0.0, value=59654.0, step=1.0)
    major_axis = st.number_input("Major Axis Length", min_value=0.0, value=351.0, step=1.0)
    minor_axis = st.number_input("Minor Axis Length", min_value=0.0, value=219.0, step=1.0)
    eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

with col2:
    convex_area = st.number_input("Convex Area", min_value=0.0, value=62384.0, step=1.0)
    extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    perimeter = st.number_input("Perimeter", min_value=0.0, value=980.0, step=1.0)

# 4. Predict button logic
if st.button("Predict Raisin Class", type="primary"):
    # Organize features into a DataFrame matching training data format
    input_data = pd.DataFrame([{
        "Area": area,
        "MajorAxisLength": major_axis,
        "MinorAxisLength": minor_axis,
        "Eccentricity": eccentricity,
        "ConvexArea": convex_area,
        "Extent": extent,
        "Perimeter": perimeter
    }])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.write("---")
    # Map the output back to human-readable names based on your notebook logic
    # (Notebook: Kecimen -> 1, Besni -> 0)
    if prediction == 1:
        st.success("🎉 The predicted class is: **Kecimen**")
    else:
        st.success("🎉 The predicted class is: **Besni**")