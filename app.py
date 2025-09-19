import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
    
import altair as alt

st.set_page_config(page_title="Dashboard DTSEN Padang Panjang", layout="wide")

st.title("📊 Dashboard Inovasi Berbasis DTSEN")
st.write("Use Case: Prediksi Kemiskinan, Prediksi Stunting, dan Clustering Hunian Kumuh")

# Load dataset hasil scoring/clustering
df = pd.read_csv("dtsen_with_scores.csv")

# Sidebar
menu = st.sidebar.radio("Pilih Use Case", [
    "Prediksi Kemiskinan",
    "Prediksi Stunting",
    "Clustering Hunian Kumuh",
    "Forecast Migrasi & Pertumbuhan Penduduk Kota"
])


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
    # st.header("🏚️ Clustering Hunian Kumuh")
    # st.write("Distribusi cluster rumah tangga")
    # cluster_count = df["cluster"].value_counts()
    # st.bar_chart(cluster_count)

    # st.write("Contoh 20 data rumah")
    # st.dataframe(df[["nik_kepala_keluarga","nama_kepala_keluarga","kelurahan","cluster"]].head(20))


    # Mapping cluster → label
    cluster_labels = {0: "Layak Huni", 1: "Semi Kumuh", 2: "Kumuh"}
    df["cluster_label"] = df["cluster"].map(cluster_labels)

    # Hitung jumlah per cluster
    cluster_count = df["cluster_label"].value_counts().reset_index()
    cluster_count.columns = ["Cluster", "Jumlah"]

    # Warna sesuai kategori
    color_scale = alt.Scale(
        domain=["Layak Huni", "Semi Kumuh", "Kumuh"],
        range=["#2ecc71", "#f1c40f", "#e74c3c"]  # hijau, kuning, merah
    )

    # Buat bar chart
    chart = alt.Chart(cluster_count).mark_bar().encode(
        x=alt.X("Cluster:N", sort=["Layak Huni","Semi Kumuh","Kumuh"]),
        y="Jumlah:Q",
        color=alt.Color("Cluster:N", scale=color_scale)
    ).properties(
        title="Distribusi Cluster Hunian"
    )

    st.altair_chart(chart, use_container_width=True)

    # Contoh data rumah
    st.write("Contoh 20 data rumah")
    st.dataframe(df[["nik_kepala_keluarga","nama_kepala_keluarga","kelurahan","cluster_label"]].head(20))

# Use Case 4: Clustering Hunian Kumuh
elif menu == "Forecast Migrasi & Pertumbuhan Penduduk Kota":

    st.header("📈 Forecast Migrasi & Pertumbuhan Penduduk Kota")

    # Load data
    hist_city = pd.read_csv("hist_penduduk_kota.csv")
    fcst_city = pd.read_csv("forecast_penduduk_kota_5y.csv")

    hist_city["period"] = pd.to_datetime(hist_city["period"])
    fcst_city["period"] = pd.to_datetime(fcst_city["period"])

    # Garis historis vs forecast
    hist_city["type"] = "Historical"
    fcst_city["type"] = "Forecast"
    fcst_city.rename(columns={"yhat":"population"}, inplace=True)

    plot_df = pd.concat([hist_city, fcst_city], ignore_index=True)

    chart = alt.Chart(plot_df).mark_line().encode(
        x="period:T",
        y="population:Q",
        color="type:N"
    ).properties(
        title="Penduduk Kota: Historis & 5 Tahun Forecast"
    )

    st.altair_chart(chart, use_container_width=True)

    # Per kelurahan (opsional)
    st.subheader("Per Kelurahan")
    kel = st.selectbox("Pilih Kelurahan", sorted(ts["kelurahan"].unique().tolist()))
    hist_k = ts[ts["kelurahan"]==kel][["date","population"]].rename(columns={"date":"period"})
    fcst_k = pd.read_csv("forecast_penduduk_prophet_5y.csv")  # atau sarimax
    fcst_k = fcst_k[fcst_k["kelurahan"]==kel][["ds","yhat"]].rename(columns={"ds":"period","yhat":"population"})
    hist_k["type"] = "Historical"; fcst_k["type"] = "Forecast"
    plot_k = pd.concat([hist_k, fcst_k], ignore_index=True)

    chart_k = alt.Chart(plot_k).mark_line().encode(
        x="period:T", y="population:Q", color="type:N"
    ).properties(title=f"Penduduk {kel}: Historis & Forecast")
    st.altair_chart(chart_k, use_container_width=True)

