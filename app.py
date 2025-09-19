import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np   
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
    "Forecast Migrasi & Pertumbuhan Penduduk Kota",
    "Segmentasi Sosial-Ekonomi",
    "Deteksi Anomali Bansos",
    "Prediksi Layanan Publik",
    "Monitoring Program Kota"
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

    # Load data kota
    hist_city = pd.read_csv("hist_penduduk_kota.csv")
    fcst_city = pd.read_csv("forecast_penduduk_kota_5y.csv")

    hist_city["period"] = pd.to_datetime(hist_city["period"])
    fcst_city["period"] = pd.to_datetime(fcst_city["period"])

    # Label tipe data
    hist_city["type"] = "Historical"
    fcst_city["type"] = "Forecast"
    fcst_city.rename(columns={"yhat":"population"}, inplace=True)

    # Gabungkan historis + forecast
    plot_df = pd.concat([hist_city, fcst_city], ignore_index=True)

    color_scale = alt.Scale(
        domain=["Historical", "Forecast"],
        range=["#2E86C1", "#E74C3C"]  # biru, merah
    )

    # Chart agregat kota
    chart = alt.Chart(plot_df).mark_line().encode(
        x="period:T",
        y="population:Q",
        color=alt.Color("type:N", scale=color_scale, title="Jenis Data")
    ).properties(
        title="Penduduk Kota: Historis & 5 Tahun Forecast"
    )
    st.altair_chart(chart, use_container_width=True)

    # chart = alt.Chart(plot_df).mark_line().encode(
    #     x="period:T", y="population:Q", color="type:N"
    # ).properties(
    #     title="Penduduk Kota: Historis & 5 Tahun Forecast"
    # )
    # st.altair_chart(chart, use_container_width=True)

    # Load data per kelurahan
    ts = pd.read_csv("ts_penduduk_kelurahan_2019_2025.csv")
    fcst_all = pd.read_csv("forecast_penduduk_prophet_5y.csv")

    # Forecast per kelurahan
    st.subheader("Per Kelurahan")
    kel = st.selectbox("Pilih Kelurahan", sorted(ts["kelurahan"].unique().tolist()))

    hist_k = ts[ts["kelurahan"]==kel][["date","population"]].rename(columns={"date":"period"})
    fcst_k = fcst_all[fcst_all["kelurahan"]==kel][["ds","yhat"]].rename(columns={"ds":"period","yhat":"population"})

    hist_k["type"] = "Historical"; fcst_k["type"] = "Forecast"
    plot_k = pd.concat([hist_k, fcst_k], ignore_index=True)

    chart_k = alt.Chart(plot_k).mark_line().encode(
        x="period:T", y="population:Q", color=alt.Color("type:N", scale=color_scale, title="Jenis Data")
    ).properties(title=f"Penduduk {kel}: Historis & Forecast")
    st.altair_chart(chart_k, use_container_width=True)

elif menu == "Segmentasi Sosial-Ekonomi":
    st.header("👨‍👩‍👧‍👦 Segmentasi Sosial-Ekonomi Wilayah")

    # Load dataset khusus segmen
    df_seg = pd.read_csv("dtsen_with_segments.csv")

    # Hitung distribusi segmen per kelurahan
    seg_per_kel = df_seg.groupby(["kelurahan","socio_segment_label"]).size().reset_index(name="jumlah")

    st.subheader("Distribusi Segmen per Kelurahan (Tabel)")
    st.dataframe(seg_per_kel)

    # 🔹 Heatmap
    st.subheader("Heatmap Segmen per Kelurahan")
    seg_pivot = seg_per_kel.pivot(index="kelurahan", columns="socio_segment_label", values="jumlah").fillna(0)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(seg_pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    # 🔹 Stacked Bar Chart
    st.subheader("Distribusi Segmen per Kelurahan (Stacked Bar)")
    seg_bar = seg_per_kel.pivot(index="kelurahan", columns="socio_segment_label", values="jumlah").fillna(0)
    seg_bar = seg_bar[["Mampu","Menengah","Rentan"]]  # urutan warna tetap

    fig, ax = plt.subplots(figsize=(10,6))
    seg_bar.plot(kind="bar", stacked=True,
                 color=["#2ecc71","#f1c40f","#e74c3c"], ax=ax)  # hijau, kuning, merah
    plt.title("Distribusi Segmen Sosial-Ekonomi per Kelurahan")
    plt.ylabel("Jumlah Keluarga")
    plt.xlabel("Kelurahan")
    st.pyplot(fig)

    # 🔹 Contoh data keluarga
    st.subheader("Contoh Data Keluarga")
    st.dataframe(df_seg[["nik_kepala_keluarga","nama_kepala_keluarga","kelurahan","socio_segment_label"]].head(20))


elif menu == "Deteksi Anomali Bansos":
    st.header("🚨 Deteksi Anomali Data Penduduk (Fraud Bansos)")

    # Load dataset dengan anomali
    df_anom = pd.read_csv("dtsen_with_anomalies.csv")

    # Ringkasan jumlah anomali
    st.subheader("Ringkasan")
    st.write(df_anom["anomaly_label"].value_counts())

    # Visualisasi
    fig, ax = plt.subplots()
    sns.countplot(x="anomaly_label", data=df_anom, palette=["#2ecc71","#e74c3c"], ax=ax)
    st.pyplot(fig)

    # Tampilkan contoh data mencurigakan
    st.subheader("Data Anomali (contoh)")
    st.dataframe(df_anom[df_anom["anomaly_label"]=="Anomali"].head(20)[
        ["nik_kepala_keluarga","nama_kepala_keluarga","pendapatan_per_bulan","penerima_bansos","anomaly_label"]
    ])

elif menu == "Prediksi Layanan Publik":
    st.header("🏥📚 Prediksi Permintaan Layanan Publik")

    # --- Ambil data real dari DTSEN ---
    df_scores = pd.read_csv("dtsen_with_scores.csv")

    # Hitung proporsi anak sekolah dari data DTSEN nyata
    proporsi_anak_sekolah = (
        df_scores["jumlah_anak_sekolah"].sum() /
        df_scores["jumlah_anggota_keluarga"].sum()
    )
    st.write(f"Proporsi anak sekolah (real dari DTSEN): {proporsi_anak_sekolah:.2%}")

    # --- Prediksi Agregat Kota ---
    fcst_city = pd.read_csv("forecast_penduduk_kota_5y.csv")
    fcst_city = fcst_city.rename(columns={"yhat":"population"})

    # Hitung kebutuhan berdasarkan proporsi nyata
    fcst_city["anak_sekolah_pred"] = (fcst_city["population"] * proporsi_anak_sekolah).round(0)
    fcst_city["puskesmas_needed"] = (fcst_city["population"] / 10000).round(0)
    fcst_city["school_needed"] = np.ceil(fcst_city["anak_sekolah_pred"] / 2000)

    st.subheader("Prediksi Agregat Kota")
    st.dataframe(fcst_city[["period","population","anak_sekolah_pred","puskesmas_needed","school_needed"]])

    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(x="period", y="puskesmas_needed", data=fcst_city, label="Puskesmas", ax=ax)
    sns.lineplot(x="period", y="school_needed", data=fcst_city, label="Sekolah", ax=ax)
    ax.set_title("Prediksi Kebutuhan Layanan Publik (Kota)")
    st.pyplot(fig)

    # --- Prediksi Per Kelurahan ---
    fcst_kel = pd.read_csv("forecast_penduduk_prophet_5y.csv")
    fcst_kel = fcst_kel.rename(columns={"ds":"period","yhat":"population"})

    fcst_kel["anak_sekolah_pred"] = (fcst_kel["population"] * proporsi_anak_sekolah).round(0)
    fcst_kel["puskesmas_needed"] = np.ceil(fcst_kel["population"] / 10000)
    fcst_kel["school_needed"] = np.ceil(fcst_kel["anak_sekolah_pred"] / 1000)  # lebih kecil kapasitasnya untuk skala kelurahan

    st.subheader("Prediksi Per Kelurahan")
    kel = st.selectbox("Pilih Kelurahan", sorted(fcst_kel["kelurahan"].unique().tolist()))

    kel_data = fcst_kel[fcst_kel["kelurahan"]==kel]

    st.dataframe(kel_data[["period","population","anak_sekolah_pred","puskesmas_needed","school_needed"]])

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.lineplot(x="period", y="puskesmas_needed", data=kel_data, label="Puskesmas", ax=ax2)
    sns.lineplot(x="period", y="school_needed", data=kel_data, label="Sekolah", ax=ax2)
    ax2.set_title(f"Prediksi Layanan Publik Kelurahan {kel}")
    st.pyplot(fig2)

elif menu == "Monitoring Program Kota":
    st.header("📊 Monitoring Dampak Program Kota")

    # Load data sebelum & sesudah
    df_before = pd.read_csv("dtsen_with_scores.csv")
    df_after = pd.read_csv("dtsen_update_2026.csv")

    # Merge berdasarkan NIK
    merged = df_before.merge(
        df_after,
        on="nik_kepala_keluarga",
        suffixes=("_before", "_after")
    )

    # Hitung perubahan skor
    merged["delta_risk"] = merged["risk_score_after"] - merged["risk_score_before"]
    merged["delta_stunting"] = merged["stunting_risk_score_after"] - merged["stunting_risk_score_before"]

    # --- Ringkasan Dampak ---
    st.subheader("Rata-rata Dampak Program")
    summary = merged[["delta_risk","delta_stunting"]].mean().reset_index()
    summary.columns = ["Indikator", "Perubahan Rata-rata"]
    st.dataframe(summary)

    # --- Histogram Perubahan Risk Score ---
    st.subheader("Distribusi Perubahan Risk Score (Before vs After)")
    fig, ax = plt.subplots()
    sns.histplot(merged["delta_risk"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # --- Top Keluarga yang Paling Membaik ---
    st.subheader("Top 20 Keluarga dengan Perbaikan Terbesar")
    top_improve = merged.sort_values("delta_risk").head(20)
    st.dataframe(top_improve[[
        "nik_kepala_keluarga","nama_kepala_keluarga","kelurahan",
        "risk_score_before","risk_score_after","delta_risk"
    ]])

    # --- Ringkasan per Kelurahan ---
    st.subheader("Dampak Program per Kelurahan")
    kel_summary = merged.groupby("kelurahan")[["delta_risk","delta_stunting"]].mean().reset_index()
    st.dataframe(kel_summary)




