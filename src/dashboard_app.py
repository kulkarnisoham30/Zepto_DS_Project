import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import traceback
from pathlib import Path

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
# @st.cache_resource
def load_model():
    """Load trained model with error handling"""
    try:
        with open("/best_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'best_model.pkl' not found. Please ensure it exists in the current directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ------------------------------
# Load Training Data (for drift analysis)
# ------------------------------
# @st.cache_data
def load_training_data():
    """Load training data for drift comparison"""
    try:
        train_data = pd.read_csv("data/final_dataset-2.csv")
        return train_data
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"Could not load training data: {str(e)}")
        return None

train_data = load_training_data()

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
                predictions = model.predict(data)
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
                    
                    explainer = shap.Explainer(model, data_sample)
                    shap_values = explainer(data_sample)
                    
                    # SHAP Summary Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, data_sample, show=False)
                    st.pyplot(fig)
                    plt.close()
                    
                    st.caption(f"SHAP analysis computed on {sample_size} samples")
                    
                except Exception as e:
                    st.warning(f"⚠️ Unable to generate SHAP plot: {str(e)}")
                    st.info("SHAP may not be compatible with this model type or data format")
        
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
            st.info("Training dataset not found at 'data/final_dataset-2.csv'. Drift analysis unavailable.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("Zepto Data Science Dashboard v1.0 | Powered by Streamlit")