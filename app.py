import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np   
import altair as alt

st.set_page_config(page_title="Dashboard DTSEN Padang Panjang", layout="wide")
col1, col2 = st.columns([1, 8])  # 1 bagian logo, 8 bagian judul

st.markdown(
    """
    <div style="display: flex; align-items: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Logopadangpanjang.png/960px-Logopadangpanjang.png?20110612034019" width="80" style="margin-right: 20px;">
        <div>
            <h1 style="margin: 0; color: #2E86C1;">Dashboard Smart City Padang Panjang</h1>
            <h3 style="margin: 0; color: #555;">Inovasi DTSEN untuk Prediksi & Perencanaan Pembangunan Kota</h3>
        </div>
    </div>
    <hr style="margin-top: 10px; margin-bottom: 20px;">
    """,
    unsafe_allow_html=True
)

# with col1:
#     st.image("logo_padangpanjang.png", width=80)  # ganti dengan path/logo Anda

# with col2:
#     st.title("Dashboard Smart City Padang Panjang")
#     st.subheader("Inovasi DTSEN untuk Prediksi & Perencanaan Pembangunan Kota")

# st.write("Use Case: Prediksi Kemiskinan, Prediksi Stunting, dan Clustering Hunian Kumuh")

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
    "Monitoring Program Kota",
    "Early Warning Krisis Ekonomi"
])


# Use Case 1: Prediksi Kemiskinan
if menu == "Prediksi Kemiskinan":
    st.header("🏠 Prediksi Kemiskinan")
    st.write("""
    **Tujuan:** Mengetahui keluarga dengan risiko kemiskinan tertinggi agar intervensi (BLT, bansos, subsidi pangan) lebih tepat sasaran.  
    **Data DTSEN:** Pendapatan per bulan, pengeluaran, pendidikan, pekerjaan, kepemilikan rumah/aset.  
    **Model:** Gradient Boosting (LightGBM) → menghasilkan *risk_score* (0–1).
    """)

    # Tambahkan status kategori
    def status_risk(score):
        if score < 0.3:
            return "Rendah"
        elif score < 0.6:
            return "Sedang"
        else:
            return "Tinggi"

    df["status_kemiskinan"] = df["risk_score"].apply(status_risk)

    # Tampilkan Top 20 keluarga rentan
    top_poor = df.sort_values("risk_score", ascending=False).head(20)
    st.subheader("Daftar 20 Keluarga dengan Risiko Kemiskinan Tertinggi")
    st.dataframe(
        top_poor[[
            "nik_kepala_keluarga",
            "nama_kepala_keluarga",
            "kelurahan",
            "kecamatan",
            "risk_score",
            "status_kemiskinan"
        ]]
    )

    # Distribusi skor
    st.subheader("Distribusi Risk Score")
    fig, ax = plt.subplots()
    sns.histplot(df["risk_score"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribusi Skor Risiko Kemiskinan")
    st.pyplot(fig)

    # Distribusi kategori
    st.subheader("Distribusi Status Risiko (Kategori)")
    st.bar_chart(df["status_kemiskinan"].value_counts())

# Use Case 2: Prediksi Stunting
elif menu == "Prediksi Stunting":
    st.header("🧒 Prediksi Stunting")
    st.write("""
    **Tujuan:** Mendeteksi anak/keluarga berisiko stunting untuk prioritas PMT, edukasi gizi, dan akses kesehatan.  
    **Data DTSEN:** Jumlah anak balita, pendidikan ibu, akses sanitasi, akses puskesmas.  
    **Model:** Gradient Boosting (LightGBM) → menghasilkan *stunting_risk_score* (0–1).
    """)

    # Tambahkan status kategori stunting
    def status_stunting(score):
        if score < 0.3:
            return "Rendah"
        elif score < 0.6:
            return "Sedang"
        else:
            return "Tinggi"

    df["status_stunting"] = df["stunting_risk_score"].apply(status_stunting)

    # Tampilkan Top 20 keluarga rentan stunting
    top_stunting = df.sort_values("stunting_risk_score", ascending=False).head(20)
    st.subheader("Daftar 20 Keluarga dengan Risiko Stunting Tertinggi")
    st.dataframe(
        top_stunting[[
            "nik_kepala_keluarga",
            "nama_kepala_keluarga",
            "kelurahan",
            "kecamatan",
            "stunting_risk_score",
            "status_stunting"
        ]]
    )

    # Distribusi skor
    st.subheader("Distribusi Risk Score Stunting")
    fig, ax = plt.subplots()
    sns.histplot(df["stunting_risk_score"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribusi Skor Risiko Stunting")
    st.pyplot(fig)

    # Distribusi kategori
    st.subheader("Distribusi Status Risiko Stunting (Kategori)")
    st.bar_chart(df["status_stunting"].value_counts())


# Use Case 3: Clustering Hunian Kumuh
elif menu == "Clustering Hunian Kumuh":
    st.header("🏚️ Clustering Hunian Kumuh")
    st.write("""
    **Tujuan:** Memetakan kondisi rumah tangga (Layak Huni, Semi Kumuh, Kumuh) sebagai dasar program bedah rumah & infrastruktur.  
    **Data DTSEN:** Jenis lantai, dinding, atap, fasilitas MCK, kepadatan rumah.  
    **Model:** K-Means Clustering → label: Layak Huni (0), Semi Kumuh (1), Kumuh (2).
    """)

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
    st.write("""
    **Tujuan:** Proyeksi jumlah penduduk 5 tahun ke depan untuk perencanaan sekolah, perumahan, dan transportasi.  
    **Data DTSEN:** Data migrasi penduduk, usia produktif, tren kelahiran.  
    **Model:** Time Series Forecasting dengan Prophet.
    """)

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
    st.write("""
    **Tujuan:** Mengelompokkan keluarga menjadi segmen Mampu, Menengah, dan Rentan agar kebijakan lebih tepat.  
    **Data DTSEN:** Pendapatan, pengeluaran, aset (lahan, kendaraan, tabungan), pendidikan, pekerjaan.  
    **Model:** K-Means Clustering → label: Mampu, Menengah, Rentan.
    """)

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
    st.write("""
    **Tujuan:** Mengidentifikasi keluarga penerima bansos yang mencurigakan (fraud/duplikasi).  
    **Data DTSEN:** NIK, pendapatan, status penerima bansos.  
    **Model:** Isolation Forest → label: Normal / Anomali.
    """)

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
    st.write("""
    **Tujuan:** Mengetahui kebutuhan Puskesmas & Sekolah berdasarkan proyeksi jumlah penduduk & proporsi anak sekolah.  
    **Data DTSEN:** Jumlah penduduk, jumlah anak sekolah, jumlah keluarga.  
    **Model:** Time Series Forecast (Prophet) + Rasio kebutuhan layanan.
    """)

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
    st.write("""
    **Tujuan:** Mengevaluasi dampak program pemerintah (bedah rumah, UMKM, bansos) terhadap skor kemiskinan & stunting.  
    **Data DTSEN:** Data keluarga sebelum & sesudah program (2025 vs 2026).  
    **Model:** Perbandingan skor risiko (before vs after) + analisis perubahan skor.
    """)


    # Load data sebelum & sesudah
    df_before = pd.read_csv("dtsen_with_scores.csv")         # ada risk_score & stunting_risk_score
    df_after = pd.read_csv("dtsen_update_2026.csv")          # sudah ada risk_score_after & stunting_risk_score_after

    # Merge berdasarkan NIK (tanpa suffixes)
    merged = df_before.merge(df_after, on="nik_kepala_keluarga")

    # Hitung perubahan skor
    merged["delta_risk"] = merged["risk_score_after"] - merged["risk_score"]
    merged["delta_stunting"] = merged["stunting_risk_score_after"] - merged["stunting_risk_score"]

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

    st.subheader("Top 20 Keluarga dengan Perbaikan Terbesar")

    top_improve = merged.sort_values("delta_risk").head(20)

    # Pilih kolom yang tersedia
    kolom_display = ["nik_kepala_keluarga","risk_score","risk_score_after","delta_risk"]
    for c in ["nama_kepala_keluarga","kelurahan"]:
        if c in merged.columns:
            kolom_display.insert(1, c)

    # Ambil data
    df_tampil = top_improve[kolom_display].copy()

    # Tambahkan kolom status
    def status_perubahan(x):
        if x < 0:
            return "Membaik"
        elif x > 0:
            return "Memburuk"
        else:
            return "Tetap"

    df_tampil["Status Perubahan"] = df_tampil["delta_risk"].apply(status_perubahan)

    # Ubah nama kolom supaya mudah dipahami
    df_tampil = df_tampil.rename(columns={
        "nik_kepala_keluarga": "NIK Kepala Keluarga",
        "nama_kepala_keluarga": "Nama Kepala Keluarga",
        "kelurahan": "Kelurahan",
        "risk_score": "Skor Kemiskinan (Sebelum Program)",
        "risk_score_after": "Skor Kemiskinan (Sesudah Program)",
        "delta_risk": "Perubahan Skor"
    })

    st.dataframe(df_tampil)


    # --- Ringkasan per Kelurahan ---
    # --- Ringkasan per Kelurahan ---
    # --- Ringkasan per Kelurahan ---
    # --- Ringkasan per Kelurahan ---
    # --- Ringkasan per Kelurahan ---
    if "kelurahan_x" in merged.columns:
        st.subheader("Dampak Program per Kelurahan")

        # rata-rata perubahan
        kel_summary = merged.groupby("kelurahan_x")[["delta_risk","delta_stunting"]].mean().reset_index()
        kel_summary = kel_summary.rename(columns={"kelurahan_x": "Kelurahan"})

        # tambahkan status per keluarga
        def status_perubahan(x):
            if x < 0:
                return "Membaik"
            elif x > 0:
                return "Memburuk"
            else:
                return "Tetap"

        merged["Status Perubahan"] = merged["delta_risk"].apply(status_perubahan)

        # hitung distribusi status per kelurahan
        kel_status = merged.groupby(["kelurahan_x","Status Perubahan"]).size().reset_index(name="Jumlah")
        kel_status_pivot = kel_status.pivot(index="kelurahan_x", columns="Status Perubahan", values="Jumlah").fillna(0).reset_index()
        kel_status_pivot = kel_status_pivot.rename(columns={"kelurahan_x": "Kelurahan"})

        # tampilkan
        st.write("📊 Rata-rata Perubahan Skor")
        st.dataframe(kel_summary)

        st.write("📊 Distribusi Status Perubahan per Kelurahan")
        st.dataframe(kel_status_pivot)

    elif "kelurahan_y" in merged.columns:
        st.subheader("Dampak Program per Kelurahan")

        kel_summary = merged.groupby("kelurahan_y")[["delta_risk","delta_stunting"]].mean().reset_index()
        kel_summary = kel_summary.rename(columns={"kelurahan_y": "Kelurahan"})

        merged["Status Perubahan"] = merged["delta_risk"].apply(status_perubahan)
        kel_status = merged.groupby(["kelurahan_y","Status Perubahan"]).size().reset_index(name="Jumlah")
        kel_status_pivot = kel_status.pivot(index="kelurahan_y", columns="Status Perubahan", values="Jumlah").fillna(0).reset_index()
        kel_status_pivot = kel_status_pivot.rename(columns={"kelurahan_y": "Kelurahan"})

        st.write("📊 Rata-rata Perubahan Skor")
        st.dataframe(kel_summary)

        st.write("📊 Distribusi Status Perubahan per Kelurahan")
        st.dataframe(kel_status_pivot)

    else:
        st.warning("Kolom kelurahan tidak ditemukan di dataset hasil merge.")

elif menu == "Early Warning Krisis Ekonomi":
    st.header("🚨 Early Warning Krisis Ekonomi Lokal")
    st.write("""
    **Tujuan:** Sistem peringatan dini bila pendapatan masyarakat turun drastis atau penerima bansos meningkat tajam.  
    **Data DTSEN:** Pendapatan per bulan, status pekerjaan, penerima bansos.  
    **Model:** Change-point detection sederhana + analisis tren 3 bulan terakhir.
    """)

    # --- Cari kolom tanggal ---
    tanggal_col = None
    for c in df.columns:
        if "tanggal_update" in c.lower():
            tanggal_col = c
            break

    if tanggal_col is None:
        st.error("❌ Tidak ada kolom 'tanggal_update' di dataset. Pastikan file CSV punya tanggal update.")
    else:
        # Pastikan datetime
        df[tanggal_col] = pd.to_datetime(df[tanggal_col], errors="coerce")

        # --- Agregasi pendapatan bulanan ---
        monthly_income = (
            df.groupby(df[tanggal_col].dt.to_period("M"))["pendapatan_per_bulan"]
            .mean()
            .reset_index()
        )
        monthly_income[tanggal_col] = monthly_income[tanggal_col].dt.to_timestamp()

        # --- Tampilkan grafik tren ---
        st.subheader("Tren Rata-rata Pendapatan Bulanan")
        st.line_chart(monthly_income.set_index(tanggal_col))

        # --- Early Warning sederhana ---
        if len(monthly_income) >= 3:
            last = monthly_income["pendapatan_per_bulan"].iloc[-1]
            prev = monthly_income["pendapatan_per_bulan"].iloc[-3]  # 3 bulan lalu

            if last < 0.8 * prev:
                st.error("⚠️ Pendapatan rata-rata turun >20% dalam 3 bulan terakhir → Potensi Krisis Ekonomi!")
            else:
                st.success("✅ Tidak ada indikasi krisis besar dalam 3 bulan terakhir.")
        else:
            st.info("ℹ️ Data belum cukup panjang untuk analisis tren (butuh minimal 3 bulan).")

        # --- Distribusi penerima bansos (indikator tambahan) ---
        st.subheader("Distribusi Penerima Bansos per Bulan")
        bansos_trend = (
            df.groupby(df[tanggal_col].dt.to_period("M"))["penerima_bansos"]
            .apply(lambda x: (x == "Ya").sum())
            .reset_index(name="jumlah_penerima")
        )
        bansos_trend[tanggal_col] = bansos_trend[tanggal_col].dt.to_timestamp()

        st.line_chart(bansos_trend.set_index(tanggal_col))

st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px; color: gray;">
        © 2025 Windi Sri Rahayu- All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)


