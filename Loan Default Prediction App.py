import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Set page configuration for a professional wide layout
st.set_page_config(
    page_title="Loan Default Risk Analytics",
    page_icon="🏦",
    layout="wide"
)

# ==================================
# Load artifacts
# ==================================

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "tuned_resnet_model.keras"
    )
    preprocessor = joblib.load(
        "preprocessor.joblib"
    )
    meta = joblib.load(
        "tuned_resnet_meta-2.joblib"
    )
    return model, preprocessor, meta

# Load components
model, preprocessor, meta = load_artifacts()
threshold = meta["threshold"]

# ==================================
# Business Rule
# ==================================

def apply_own_car_rule(df):
    df = df.copy()
    df.loc[
        (df["FLAG_OWN_CAR"] == "N")
        & (df["OWN_CAR_AGE"].notnull()),
        "OWN_CAR_AGE"
    ] = 0
    return df

# ==================================
# UI / Header
# ==================================

st.title("🏦 Loan Default Prediction Analytics")
st.markdown(
    "Upload a customer portfolio CSV file below to instantly evaluate credit risk and default probabilities."
)
st.write("---")

# ==================================
# Data Input Section
# ==================================

# Initialize a session state variable to track if the sample button was clicked
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

col_upload, col_sample = st.columns([3, 1], vertical_alignment="bottom")

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose a customer data file (CSV format)",
        type=["csv"],
        help="Ensure columns match the training dataset schema before uploading."
    )

with col_sample:
    # Button to instantly load the repository file
    if st.button("✨ Try with a Sample", use_container_width=True):
        st.session_state.use_sample = True

# Determine which data source to use
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.use_sample = False  # Reset sample flag if they upload a file
elif st.session_state.use_sample:
    try:
        # Make sure "sample.csv" matches your file name in GitHub exactly
        df = pd.read_csv("loan_trial.csv") 
    except FileNotFoundError:
        st.error("⚠️ 'sample.csv' could not be found in your repository root.")

# ==================================
# Prediction Pipeline Processing
# ==================================

if df is not None:
    
    with st.expander("📄 View Uploaded Raw Data Preview", expanded=False):
        st.dataframe(df, use_container_width=True)

    try:
        # Preprocessing & Predictions
        df = apply_own_car_rule(df)
        X = preprocessor.transform(df)

        probs = model.predict(X, verbose=0).flatten()
        preds = (probs >= threshold).astype(int)

        # Build results DataFrame
        result = df.copy()
        result["Default_Probability"] = probs
        result["Prediction"] = np.where(
            preds == 1,
            "Risky",
            "Pays Regularly"
        )

        # Move critical target columns to the front
        front_cols = ["Default_Probability", "Prediction"]
        other_cols = [col for col in result.columns if col not in front_cols]
        result = result[front_cols + other_cols]

        st.success("🎉 Prediction Pipeline Executed Successfully!")
        st.write("---")

        # ==================================
        # Executive Summary Metrics Dashboard
        # ==================================
        st.subheader("📊 Portfolio Risk Summary")
        
        total_records = len(result)
        risky_count = int(np.sum(preds == 1))
        avg_prob = float(np.mean(probs))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Records Processed", value=f"{total_records:,}")
        with col2:
            st.metric(
                label="High Risk Applications Detected", 
                value=f"{risky_count:,}",
                delta=f"{(risky_count/total_records)*100:.1f}% of portfolio",
                delta_color="inverse"
            )
        with col3:
            st.metric(label="Average Default Probability", value=f"{avg_prob:.2%}")

        st.write("---")

        # ==================================
        # Detailed Analytics Section
        # ==================================
        st.subheader("🔍 Detailed Risk Assessment")
        
        dl_col, space_col = st.columns([1, 4])
        with dl_col:
            st.download_button(
                label="📥 Download Results CSV",
                data=result.to_csv(index=False),
                file_name="loan_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.dataframe(
            result.style.format({"Default_Probability": "{:.2%}"})
            .map(
                lambda val: "background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold;" if val == "Risky" else "",
                subset=["Prediction"]
            ),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"An error occurred during parsing: {str(e)}")
