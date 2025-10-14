# dashboard_app.py
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Zepto DS Dashboard", layout="wide")
st.title("📊 Zepto Data Science Dashboard")
st.markdown("### Predictive Insights • Model Explainability • Responsible AI View")

# Sidebar
st.sidebar.header("Upload Data for Prediction")
uploaded = st.sidebar.file_uploader("data/final dataset-2.csv", type=["csv"])

# ------------------------------
# Load and Predict
# ------------------------------
if uploaded:
    data = pd.read_csv(uploaded)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        preds = model.predict(data)
        st.write("### 🔮 Predictions")
        data["Predicted"] = preds
        st.dataframe(data.head())

        # Download predictions
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ------------------------------
# SHAP Explainability
# ------------------------------
st.write("---")
st.subheader("🔍 Model Explainability using SHAP")

if uploaded:
    try:
        explainer = shap.Explainer(model, data)
        shap_values = explainer(data)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, data, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Unable to generate SHAP plot. Ensure the data matches model input.")
else:
    st.info("Upload a dataset to view SHAP explainability results.")

# ------------------------------
# Drift Check (Basic Example)
# ------------------------------
st.write("---")
st.subheader("📈 Data Drift Check")

try:
    train_data = pd.read_csv("data/final_dataset-2.csv")
    drift = {}

    for col in train_data.columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(train_data[col]):
            drift[col] = abs(train_data[col].mean() - data[col].mean())

    drift_df = pd.DataFrame.from_dict(drift, orient="index", columns=["Mean Drift"])
    st.dataframe(drift_df)
except Exception as e:
    st.warning("Drift check not available — ensure training dataset exists.")
