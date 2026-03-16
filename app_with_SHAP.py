import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💊",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS Styling
# -------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
    text-align: center;
}

.prediction-box {
    padding:20px;
    border-radius:10px;
    background-color:#e8f4ff;
    font-size:28px;
    text-align:center;
    color:#0b5394;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = pickle.load(open("best_model.pkl","rb"))

# Feature names used during training
features = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region_northwest",
    "region_southeast",
    "region_southwest"
]

# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("💊 Medical Insurance Cost Prediction")

st.write(
"""
This ML application predicts **medical insurance charges** based on patient data.
The model was trained using **Random Forest Regression**.
"""
)

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age",18,100,30)

sex = st.sidebar.selectbox(
    "Sex",
    ["male","female"]
)

bmi = st.sidebar.slider(
    "BMI",
    10.0,
    50.0,
    25.0
)

children = st.sidebar.slider(
    "Children",
    0,
    5,
    0
)

smoker = st.sidebar.selectbox(
    "Smoker",
    ["yes","no"]
)

region = st.sidebar.selectbox(
    "Region",
    ["northeast","northwest","southeast","southwest"]
)

predict_btn = st.sidebar.button("Predict Insurance Cost")

# -------------------------------------------------
# Data Processing
# -------------------------------------------------

sex = 1 if sex=="male" else 0
smoker = 1 if smoker=="yes" else 0

region_northwest = 0
region_southeast = 0
region_southwest = 0

if region == "northwest":
    region_northwest = 1
elif region == "southeast":
    region_southeast = 1
elif region == "southwest":
    region_southwest = 1

input_data = pd.DataFrame([[
    age,
    sex,
    bmi,
    children,
    smoker,
    region_northwest,
    region_southeast,
    region_southwest
]], columns=features)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if predict_btn:

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class="prediction-box">
        Predicted Insurance Cost: ${prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # -------------------------------------------------
    # SHAP Explainability
    # -------------------------------------------------

    st.subheader("🔎 Model Explanation (SHAP)")

# Create SHAP explainer
explainer = shap.Explainer(model)

# Generate SHAP values for the input
shap_values = explainer(input_data)

# Create plot
fig, ax = plt.subplots()

shap.plots.waterfall(shap_values[0], show=False)

# Display in Streamlit
st.pyplot(fig)
