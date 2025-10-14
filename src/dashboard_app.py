import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import traceback
from pathlib import Path
import mlflow
import mlflow.sklearn

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Zepto DS Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    """Load trained model with error handling"""
    try:
        with open("best_model.pkl", "rb") as file:
            mlflow_model = pickle.load(file)
        
        # Extract the actual sklearn model from MLflow wrapper
        if hasattr(mlflow_model, '_model_impl'):
            # MLflow PyFunc model - get the underlying implementation
            underlying = mlflow_model._model_impl
            if hasattr(underlying, 'sklearn_model'):
                return underlying.sklearn_model, mlflow_model
            return underlying, mlflow_model
        elif hasattr(mlflow_model, 'sklearn_model'):
            return mlflow_model.sklearn_model, mlflow_model
        
        # Not an MLflow model
        return mlflow_model, mlflow_model
    except FileNotFoundError:
        st.error("❌ Model file 'best_model.pkl' not found. Please ensure it exists in the current directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

sklearn_model, mlflow_model = load_model()

# ------------------------------
# Load Training Data (for drift analysis)
# ------------------------------
@st.cache_data
def load_training_data():
    """Load training data for drift comparison"""
    # Check for .pkl files first, then .csv
    possible_paths = [
        ("data/final_dataset-2.pkl", "pkl"),
        ("final_dataset-2.pkl", "pkl"),
        ("data/training_data.pkl", "pkl"),
        ("training_data.pkl", "pkl"),
        ("data/final_dataset-2.csv", "csv"),
        ("final_dataset-2.csv", "csv"),
        ("data/final dataset-2.csv", "csv"),
        ("final dataset-2.csv", "csv")
    ]
    
    for path, file_type in possible_paths:
        try:
            if Path(path).exists():
                if file_type == "pkl":
                    with open(path, "rb") as f:
                        train_data = pickle.load(f)
                        # Convert to DataFrame if it's a numpy array or other format
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
# Main UI
# ------------------------------
st.title("📊 Zepto Data Science Dashboard")
st.markdown("### Predictive Insights • Model Explainability • Responsible AI View")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# File Upload
st.sidebar.subheader("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file for prediction",
    type=["csv"],
    help="Upload a CSV file with features matching the model's requirements"
)

# Options
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Options")
show_raw_data = st.sidebar.checkbox("Show raw uploaded data", value=True)
show_shap = st.sidebar.checkbox("Show SHAP explainability", value=True)
show_drift = st.sidebar.checkbox("Show drift analysis", value=True)
max_display_rows = st.sidebar.slider("Max rows to display", 5, 100, 10)

# ------------------------------
# Main Content Area
# ------------------------------
if uploaded_file is None:
    # Welcome Screen
    st.info("👈 Please upload a CSV file from the sidebar to get started")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔮 Predictions")
        st.write("Get instant predictions on your uploaded data using our trained ML model")
    
    with col2:
        st.markdown("### 🔍 Explainability")
        st.write("Understand model decisions with SHAP values and feature importance")
    
    with col3:
        st.markdown("### 📈 Drift Detection")
        st.write("Monitor data distribution changes compared to training data")
    
else:
    # Process uploaded file
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        
        # Display raw data
        if show_raw_data:
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(data.head(max_display_rows), use_container_width=True)
            st.caption(f"Showing {min(max_display_rows, len(data))} of {len(data)} rows")
        
        st.markdown("---")
        
        # Make Predictions
        st.subheader("🔮 Predictions")
        
        with st.spinner("Generating predictions..."):
            try:
                # Use MLflow model for predictions (it has proper predict interface)
                predictions = mlflow_model.predict(data)
                data_with_preds = data.copy()
                data_with_preds["Predicted"] = predictions
                
                # Display predictions
                st.success(f"✅ Successfully generated {len(predictions)} predictions")
                st.dataframe(data_with_preds.head(max_display_rows), use_container_width=True)
                
                # Download button
                csv = data_with_preds.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Prediction statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(predictions))
                with col2:
                    st.metric("Mean Prediction", f"{np.mean(predictions):.4f}")
                with col3:
                    st.metric("Std Dev", f"{np.std(predictions):.4f}")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.code(traceback.format_exc())
        
        # SHAP Explainability
        if show_shap:
            st.markdown("---")
            st.subheader("🔍 Model Explainability using SHAP")
            
            with st.spinner("Generating SHAP values..."):
                try:
                    # Limit data for SHAP to avoid performance issues
                    sample_size = min(100, len(data))
                    data_sample = data.head(sample_size)
                    
                    # Use the sklearn model (not MLflow wrapper) for SHAP
                    st.info(f"Model type: {type(sklearn_model).__name__}")
                    
                    # Try different SHAP explainers based on model type
                    explainer = None
                    shap_values = None
                    
                    model_type = type(sklearn_model).__name__
                    
                    # Tree-based models
                    if any(x in model_type.lower() for x in ['xgb', 'lightgbm', 'gbm', 'forest', 'tree']):
                        st.info("Using TreeExplainer for tree-based model...")
                        explainer = shap.TreeExplainer(sklearn_model)
                        shap_values = explainer.shap_values(data_sample)
                    
                    # Linear models
                    elif any(x in model_type.lower() for x in ['linear', 'logistic', 'ridge', 'lasso']):
                        st.info("Using LinearExplainer for linear model...")
                        explainer = shap.LinearExplainer(sklearn_model, data_sample)
                        shap_values = explainer.shap_values(data_sample)
                    
                    # Fallback to KernelExplainer
                    else:
                        st.info("Using KernelExplainer (this may take a moment)...")
                        background = shap.sample(data_sample, min(50, len(data_sample)))
                        
                        # Create prediction function that works with the sklearn model
                        def model_predict(X):
                            return sklearn_model.predict(X)
                        
                        explainer = shap.KernelExplainer(model_predict, background)
                        shap_values = explainer.shap_values(data_sample)
                    
                    # SHAP Summary Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if isinstance(shap_values, list):
                        # Multi-class classification
                        shap.summary_plot(shap_values[0], data_sample, show=False)
                    else:
                        shap.summary_plot(shap_values, data_sample, show=False)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    st.success(f"✅ SHAP analysis completed on {sample_size} samples")
                    
                except Exception as e:
                    st.error(f"❌ Unable to generate SHAP plot: {str(e)}")
                    with st.expander("🐛 Debug Information"):
                        st.code(traceback.format_exc())
                        st.write("Model attributes:", dir(sklearn_model))
                    st.info("💡 Try uploading a smaller dataset or contact support.")
        
        # Drift Analysis
        if show_drift and train_data is not None:
            st.markdown("---")
            st.subheader("📈 Data Drift Analysis")
            
            try:
                drift_results = {}
                
                for col in train_data.columns:
                    if col in data.columns and pd.api.types.is_numeric_dtype(train_data[col]):
                        train_mean = train_data[col].mean()
                        current_mean = data[col].mean()
                        drift = abs(train_mean - current_mean)
                        drift_pct = (drift / abs(train_mean) * 100) if train_mean != 0 else 0
                        
                        drift_results[col] = {
                            "Training Mean": train_mean,
                            "Current Mean": current_mean,
                            "Absolute Drift": drift,
                            "Drift %": drift_pct
                        }
                
                if drift_results:
                    drift_df = pd.DataFrame.from_dict(drift_results, orient="index")
                    drift_df = drift_df.round(4)
                    drift_df = drift_df.sort_values("Drift %", ascending=False)
                    
                    st.dataframe(drift_df, use_container_width=True)
                    
                    # Highlight high drift features
                    high_drift = drift_df[drift_df["Drift %"] > 10]
                    if not high_drift.empty:
                        st.warning(f"⚠️ {len(high_drift)} feature(s) show >10% drift from training data")
                        st.dataframe(high_drift, use_container_width=True)
                    else:
                        st.success("✅ No significant drift detected")
                else:
                    st.info("No numeric columns found for drift analysis")
                    
            except Exception as e:
                st.warning(f"⚠️ Drift analysis failed: {str(e)}")
        
        elif show_drift and train_data is None:
            st.markdown("---")
            st.subheader("📈 Data Drift Analysis")
            st.info("⚠️ Training dataset not found. Please place your training data file in one of these locations:")
            st.code("\n".join([
                "PKL files (preferred):",
                "• data/best_model.pkl",
                "• best_model.pkl",
                "• data/best_model.pkl",
                "• best_model.pkl",
                "",
                "CSV files:",
                "• data/final_dataset-2.csv",
                "• final_dataset-2.csv"
            ]))
        
        else:
            if train_data is not None and train_data_path:
                st.sidebar.success(f"✅ Training data loaded from: {train_data_path}")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("Zepto Data Science Dashboard v1.0 | Powered by Streamlit")
