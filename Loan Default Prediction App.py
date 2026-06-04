import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

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
# UI
# ==================================

st.title("Loan Default Prediction")

tab1, tab2 = st.tabs(
    ["Upload CSV", "Manual Input"]
)

# ==================================
# TAB 1
# ==================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload customer data",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.write(df.head())

        try:

            df = apply_own_car_rule(df)

            X = preprocessor.transform(df)

            probs = model.predict(
                X,
                verbose=0
            ).flatten()

            preds = (
                probs >= threshold
            ).astype(int)

            result = df.copy()

            result["Default_Probability"] = probs

            result["Prediction"] = np.where(
                preds == 1,
                "Risky",
                "Pays Regularly"
            )

            st.success("Prediction completed")

            st.dataframe(result)

            st.download_button(
                "Download Results",
                result.to_csv(index=False),
                "predictions.csv"
            )

        except Exception as e:

            st.error(str(e))
