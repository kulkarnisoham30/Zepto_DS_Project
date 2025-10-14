# dashboard_app.py
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import traceback
from pathlib import Path

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_model():
    """Load the trained model from pickle file"""
    try:
        with open("best_model.pkl", "rb") as file:
            mlflow_model = pickle.load(file)
        
        # Extract sklearn model from MLflow wrapper if needed
        if hasattr(mlflow_model, '_model_impl'):
            underlying = mlflow_model._model_impl
            if hasattr(underlying, 'sklearn_model'):
                return underlying.sklearn_model, mlflow_model
            return underlying, mlflow_model
        elif hasattr(mlflow_model, 'sklearn_model'):
            return mlflow_model.sklearn_model, mlflow_model
        
        return mlflow_model, mlflow_model
    except FileNotFoundError:
        st.error("❌ Model file 'best_model.pkl' not found")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

sklearn_model, mlflow_model = load_model()

# ------------------------------
# Load Training Data
# ------------------------------
@st.cache_data
def load_training_data():
    """Load training data for drift analysis"""
    possible_paths = [
        "data/final_dataset-2.pkl",
        "final_dataset-2.pkl",
        "data/final_dataset-2.csv",
        "final_dataset-2.csv",
        "data/final dataset-2.csv",
        "final dataset-2.csv"
    ]
    
    for path in possible_paths:
        try:
            if Path(path).exists():
                if path.endswith('.pkl'):
                    with open(path, "rb") as f:
                        train_data = pickle.load(f)
                        if not isinstance(train_data, pd.DataFrame):
                            train_data = pd.DataFrame(train_data)
                else:
                    train_data = pd.read_csv(path)
                return train_data, path
        except Exception as e:
            continue
    
    return None, None

train_data, train_data_path = load_training_data()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Zepto DS Dashboard", layout="wide")
st.title("📊 Zepto Data Science Dashboard")
st.markdown("### Predictive Insights • Model Explainability • Responsible AI View")

# Sidebar
st.sidebar.header("Upload Data for Prediction")
uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if train_data is not None:
    st.sidebar.success(f"✅ Training data loaded from: {train_data_path}")
else:
    st.sidebar.warning("⚠️ Training data not found - drift analysis unavailable")

# ------------------------------
# Load and Predict
# ------------------------------
if uploaded:
    data = pd.read_csv(uploaded)
    
    st.write("### 📄 Uploaded Data Preview")
    st.dataframe(data.head(), use_container_width=True)
    st.caption(f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
    
    # ------------------------------
    # Predictions
    # ------------------------------
    try:
        preds = mlflow_model.predict(data)
        
        st.write("### 🔮 Predictions")
        data["Predicted"] = preds
        st.dataframe(data.head(), use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(preds))
        with col2:
            st.metric("Mean Prediction", f"{np.mean(preds):.4f}")
        with col3:
            st.metric("Std Dev", f"{np.std(preds):.4f}")
        
        # Download predictions
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Predictions as CSV",
            csv,
            "predictions.csv",
            "text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        with st.expander("View Error Details"):
            st.code(traceback.format_exc())
    
    # ------------------------------
    # SHAP Explainability
    # ------------------------------
    st.write("---")
    st.subheader("🔍 Model Explainability using SHAP")
    
    if uploaded:
        try:
            # Limit data for performance
            sample_size = min(100, len(data))
            data_sample = data.head(sample_size)
            
            # Remove the Predicted column if it exists
            if 'Predicted' in data_sample.columns:
                data_sample = data_sample.drop('Predicted', axis=1)
            
            st.info(f"Computing SHAP values for {sample_size} samples...")
            
            # Determine model type and use appropriate explainer
            model_type = type(sklearn_model).__name__
            st.caption(f"Model type: {model_type}")
            
            # Try different explainers
            if any(x in model_type.lower() for x in ['xgb', 'lightgbm', 'gbm', 'forest', 'tree']):
                explainer = shap.TreeExplainer(sklearn_model)
                shap_values = explainer.shap_values(data_sample)
            elif any(x in model_type.lower() for x in ['linear', 'logistic', 'ridge', 'lasso']):
                explainer = shap.LinearExplainer(sklearn_model, data_sample)
                shap_values = explainer.shap_values(data_sample)
            else:
                # Fallback to KernelExplainer
                background = shap.sample(data_sample, min(50, len(data_sample)))
                explainer = shap.KernelExplainer(sklearn_model.predict, background)
                shap_values = explainer.shap_values(data_sample)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], data_sample, show=False)
            else:
                shap.summary_plot(shap_values, data_sample, show=False)
            st.pyplot(fig)
            plt.close()
            
            st.success("✅ SHAP analysis completed")
            
        except Exception as e:
            st.warning(f"⚠️ Unable to generate SHAP plot: {str(e)}")
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())
    else:
        st.info("Upload a dataset to view SHAP explainability results.")
    
    # ------------------------------
    # Drift Check
    # ------------------------------
    st.write("---")
    st.subheader("📈 Data Drift Check")
    
    if train_data is not None:
        try:
            drift = {}
            
            # Remove Predicted column from comparison
            data_for_drift = data.drop('Predicted', axis=1) if 'Predicted' in data.columns else data
            
            for col in train_data.columns:
                if col in data_for_drift.columns and pd.api.types.is_numeric_dtype(train_data[col]):
                    train_mean = train_data[col].mean()
                    current_mean = data_for_drift[col].mean()
                    drift_value = abs(train_mean - current_mean)
                    drift_pct = (drift_value / abs(train_mean) * 100) if train_mean != 0 else 0
                    
                    drift[col] = {
                        "Training Mean": train_mean,
                        "Current Mean": current_mean,
                        "Absolute Drift": drift_value,
                        "Drift %": drift_pct
                    }
            
            if drift:
                drift_df = pd.DataFrame.from_dict(drift, orient="index")
                drift_df = drift_df.round(4)
                drift_df = drift_df.sort_values("Drift %", ascending=False)
                
                st.dataframe(drift_df, use_container_width=True)
                
                # Highlight high drift
                high_drift = drift_df[drift_df["Drift %"] > 10]
                if not high_drift.empty:
                    st.warning(f"⚠️ {len(high_drift)} feature(s) show >10% drift from training data")
                    st.dataframe(high_drift, use_container_width=True)
                else:
                    st.success("✅ No significant drift detected (all features <10% drift)")
            else:
                st.info("No numeric columns available for drift analysis")
                
        except Exception as e:
            st.warning(f"⚠️ Drift check failed: {str(e)}")
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())
    else:
        st.info("⚠️ Training dataset not found. Place your training data at one of these locations:")
        st.code("\n".join([
            "• data/final_dataset-2.pkl",
            "• data/final_dataset-2.csv",
            "• final_dataset-2.csv"
        ]))

else:
    # Welcome screen
    st.info("👈 Upload a CSV file from the sidebar to get started")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔮 Predictions")
        st.write("Get instant predictions on your uploaded data")
    
    with col2:
        st.markdown("### 🔍 Explainability")
        st.write("Understand model decisions with SHAP values")
    
    with col3:
        st.markdown("### 📈 Drift Detection")
        st.write("Monitor data distribution changes")

# Footer
st.markdown("---")
st.caption("Zepto Data Science Dashboard | Powered by Streamlit")
