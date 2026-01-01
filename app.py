import streamlit as st
import pandas as pd
#import joblib
import cloudpickle
# ==============================
#   HEADER & UI
# ==============================
st.image("http://www.ehtp.ac.ma/images/lo.png")
st.write("""
## MSDE6 : ML Course
### IoT IDS Prediction App
This app predicts **Attack Classes** based on IoT traffic data
""")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1048/1048949.png", width=250)

st.sidebar.write("### Load Excel File for Prediction")

uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

# ==============================
#   LOAD MODEL + ENCODER
# ==============================
@st.cache_resource
def load_pipeline():
    import cloudpickle
    with open("model.pkl", "rb") as f:
        pipeline = cloudpickle.load(f)
    with open("target.pkl", "rb") as f:
        label_encoder = cloudpickle.load(f)
    return pipeline, label_encoder

# ==============================
#   PROCESS FILE
# ==============================
if uploaded_file is not None:
    # Read user Excel file
    RT_IOT2022= pd.read_excel(uploaded_file)

    st.subheader("📄 Input Data Preview")
    st.write(RT_IOT2022.head())

    # Predict
    predictions = pipeline.predict(RT_IOT2022)

    # Decode labels
    decoded_predictions = label_encoder.inverse_transform(predictions)

    # Probabilities
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        probs = pipeline.predict_proba(RT_IOT2022)
        proba_RT_IOT2022 = pd.DataFrame(probs, columns=label_encoder.classes_)
    else:
        proba_df = None

    # ==============================
    #   DISPLAY RESULTS
    # ==============================
    st.subheader("🎯 Predicted Class")
    st.write(decoded_predictions)

    if proba_RT_IOT2022 is not None:
        st.subheader("📊 Prediction Probabilities")
        st.write(proba_RT_IOT2022)

else:
    st.write("➡️ Please upload an Excel (.xlsx) file to start prediction.")