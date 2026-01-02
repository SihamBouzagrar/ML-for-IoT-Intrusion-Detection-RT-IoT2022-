import streamlit as st
import pandas as pd
import pickle

# ==============================
#   HEADER & UI
# ==============================
import sklearn
st.write("scikit-learn version:", sklearn.__version__)



st.image("http://www.ehtp.ac.ma/images/lo.png")
st.write("""

### IoT IDS Prediction App
This app predicts **Attack Classes** based on IoT traffic data
""")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1048/1048949.png", width=250)

st.sidebar.write("### Load Excel File for Prediction")

uploaded_file = st.sidebar.file_uploader(
    "Upload your Excel file",
    type=["xlsx"]
)

# ==============================
#   LOAD MODEL + ENCODER
# ==============================


@st.cache_resource
def load_pipeline():
    try:
        with open("pipeline1.pkl", "rb") as f:
            pipeline1 = pickle.load(f)
        with open("final_model1.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        return pipeline1, label_encoder
    except Exception as e:
        st.error(f"Erreur lors du chargement des fichiers pickle : {e}")
        raise e
pipeline, label_encoder = load_pipeline()

# ==============================
#   PROCESS FILE
# ==============================

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Input Data Preview")
    st.write(df.head())

    # ---- Prediction ----
    predictions = pipeline.predict(df)

    # ---- Decode labels ----
    decoded_predictions = label_encoder.inverse_transform(predictions)

    st.subheader("🎯 Predicted Class")
    st.write(decoded_predictions)

    # ---- Probabilities (if available) ----
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        probs = pipeline.predict_proba(df)
        proba_df = pd.DataFrame(probs, columns=label_encoder.classes_)

        st.subheader("📊 Prediction Probabilities")
        st.write(proba_df)

else:
    st.write("➡️ Please upload an Excel (.xlsx) file to start prediction.")