import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import sklearn
import plotly.express as px

# ==============================
# CONFIGURATION DE LA PAGE
# ==============================
st.set_page_config(
    page_title="IoT Intrusion Detection (RT-IoT2022)",
    page_icon="🛡️",
    layout="wide"
)

# ==============================
# CHARGEMENT DU MODÈLE
# ==============================
@st.cache_resource
def load_pipeline():
    with open("pipeline1.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("final_model1.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return pipeline, label_encoder

pipeline, label_encoder = load_pipeline()

# ==============================
# HEADER PRINCIPAL
# ==============================
col1, col2 = st.columns([1, 3])  # ajuster la proportion
with col2:
# Titre et sous-titre
    st.title("🛡️ Système de Détection d'Intrusions IoT")
  

st.markdown("""
    ### Mini-projet ML
    Cette application utilise des **modèles de Machine Learning** pour détecter
    les attaques réseau dans des environnements **IoT** à partir du dataset **RT-IoT2022**.
    """)
  

# ==============================
# DICTIONNAIRE DES CLASSES
# ==============================
attack_type_dict = {
    'ARP_poisioning 🖧': 0,
    'DDOS_Slowloris 💥': 1,
    'DOS_SYN_Hping ⚡': 2,
    'MQTT_Publish 📡': 3,
    'Metasploit_Brute_Force_SSH 🔐': 4,
    'NMAP_FIN_SCAN 🕵️‍♂️': 5,
    'NMAP_OS_DETECTION 🖥️': 6,
    'NMAP_TCP_scan 🔎': 7,
    'NMAP_UDP_SCAN 🧭': 8,
    'NMAP_XMAS_TREE_SCAN 🎄': 9,
    'Thing_Speak 🌐': 10,
    'Wipro_bulb 💡': 11
}

           
st.markdown("### 🔍 Signification des classes **Attack_type**")
for attack, code in attack_type_dict.items():
    st.markdown(f"- **{attack}** : code = `{code}`")

st.divider()

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcShFS5Aos0PhDsLhfPJL6Irlm3GqgHD6bCCZg&s",
        width=250
    )
    st.header("📥 Chargement des données")
    uploaded_file = st.file_uploader(
        "Uploader un fichier CSV ou Excel",
        type=["csv", "xlsx"]
    )
    st.divider()
    st.subheader("🎓 Contexte Académique")
    st.info("""
    **Réalisée par :** Siham Bouzagrar  
    **Module :** Machine Learning / Data Science  
    **Encadrant :** Mr. Abdelhamid FADIL  
    
    """)

# ==============================
# TRAITEMENT DU FICHIER
# ==============================
if uploaded_file is not None:
    try:
        # --- Lecture du fichier ---
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # --- Spinner & Progress Bar ---
        with st.spinner('Analyse du flux réseau en cours...'):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)

        # ==============================
        # STATISTIQUES
        # ==============================
        st.subheader("📊 Statistiques des données chargées")
        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre d’instances", df.shape[0])
        col2.metric("Nombre de caractéristiques", df.shape[1])
        col3.metric("Type de classification", "Multi-classe")
        st.divider()

        # ==============================
        # APERÇU DES DONNÉES
        # ==============================
        st.subheader("📄 Aperçu des données")
        st.dataframe(df.head())
        st.divider()

        # ==============================
        # PRÉDICTION
        # ==============================
        st.subheader("🎯 Résultats de la prédiction")
        predictions = pipeline.predict(df)
        decoded_predictions = label_encoder.inverse_transform(predictions)

        st.success("✅ L'analyse des intrusions est terminée avec succès !")
        st.write("### Classe(s) prédite(s)")
        st.write(decoded_predictions)

        st.balloons()

        # ==============================
        # PROBABILITÉS
        # ==============================
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            st.subheader("📊 Probabilités de prédiction")
            probs = pipeline.predict_proba(df)
            proba_df = pd.DataFrame(probs, columns=label_encoder.classes_)
            st.dataframe(proba_df)

        
          
           
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du fichier : {e}")

else:
    st.info("➡️ Veuillez charger un fichier CSV ou Excel pour lancer la prédiction.")
