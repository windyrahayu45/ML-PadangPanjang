import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard DTSEN Padang Panjang", layout="wide")

st.title("📊 Dashboard Inovasi Berbasis DTSEN")
st.write("Use Case: Prediksi Kemiskinan, Prediksi Stunting, dan Clustering Hunian Kumuh")

# Load dataset hasil scoring/clustering
df = pd.read_csv("dtsen_with_scores.csv")

# Sidebar
menu = st.sidebar.radio("Pilih Use Case", ["Prediksi Kemiskinan", "Prediksi Stunting", "Clustering Hunian Kumuh"])

# Use Case 1: Prediksi Kemiskinan
if menu == "Prediksi Kemiskinan":
    st.header("🏠 Prediksi Kemiskinan")
    st.write("Daftar keluarga dengan skor risiko kemiskinan tertinggi")
    top_poor = df.sort_values("risk_score", ascending=False).head(20)
    st.dataframe(top_poor[["nik_kepala_keluarga","nama_kepala_keluarga","kelurahan","kecamatan","risk_score"]])
    
    st.subheader("Distribusi Risk Score")
    fig, ax = plt.subplots()
    sns.histplot(df["risk_score"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# Use Case 2: Prediksi Stunting
elif menu == "Prediksi Stunting":
    st.header("🧒 Prediksi Stunting")
    st.write("Daftar keluarga dengan skor risiko stunting tertinggi")
    top_stunting = df.sort_values("stunting_risk_score", ascending=False).head(20)
    st.dataframe(top_stunting[["nik_kepala_keluarga","nama_kepala_keluarga","kelurahan","kecamatan","stunting_risk_score"]])
    
    st.subheader("Distribusi Risk Score")
    fig, ax = plt.subplots()
    sns.histplot(df["stunting_risk_score"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# Use Case 3: Clustering Hunian Kumuh
elif menu == "Clustering Hunian Kumuh":
    st.header("🏚️ Clustering Hunian Kumuh")
    st.write("Distribusi cluster rumah tangga")
    cluster_count = df["cluster"].value_counts()
    st.bar_chart(cluster_count)

    st.write("Contoh 20 data rumah")
    st.dataframe(df[["nik_kepala_keluarga","nama_kepala_keluarga","kelurahan","cluster"]].head(20))
