import streamlit as st
import pandas as pd
import cloudpickle

# ==============================
#   HEADER & UI
# ==============================
st.set_page_config(page_title="IoT IDS Prediction", layout="wide")

st.image("http://www.ehtp.ac.ma/images/lo.png")
st.write("""
## MSDE6 : ML Course
### IoT IDS Prediction App
This app predicts **Attack Classes** based on IoT traffic data
""")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1048/1048949.png", width=250)
st.sidebar.write("### Load CSV or Excel File for Prediction")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

# ==============================
#   LOAD MODEL + ENCODER
# ==============================
@st.cache_resource
def load_pipeline():
    with open("model.pkl", "rb") as f:
        pipeline = cloudpickle.load(f)
    with open("target.pkl", "rb") as f:
        label_encoder = cloudpickle.load(f)
    return pipeline, label_encoder

pipeline, label_encoder = load_pipeline()

# ==============================
#   PROCESS FILE
# ==============================
if uploaded_file is not None:
    # Detect file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:  # Excel
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    
    st.subheader("📄 Input Data Preview")
    st.dataframe(df.head())

    # Predict
    try:
        predictions = pipeline.predict(df)
    except ValueError as e:
        st.error(f"Error: {e}")
        st.stop()

    # Decode labels
    decoded_predictions = label_encoder.inverse_transform(predictions)

    # Probabilities (si le modèle les fournit)
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        probs = pipeline.predict_proba(df)
        proba_df = pd.DataFrame(probs, columns=label_encoder.classes_)
    else:
        proba_df = None

    # ==============================
    #   DISPLAY RESULTS
    # ==============================
    st.subheader("🎯 Predicted Class")
    st.dataframe(decoded_predictions)

    if proba_df is not None:
        st.subheader("📊 Prediction Probabilities")
        st.dataframe(proba_df)

else:
    st.info("➡️ Please upload a CSV or Excel file to start prediction.")
