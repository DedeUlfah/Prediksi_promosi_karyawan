import streamlit as st
import pandas as pd
import joblib

# Load model dan fitur
model = joblib.load(r'C:\Users\ASUS\Documents\miniprojek\employee\model_employee_promotion.pkl')
fitur = joblib.load(r'C:\Users\ASUS\Documents\miniprojek\employee\fitur_employee_promotion.pkl')

st.set_page_config(page_title="Prediksi Promosi", layout="centered")
st.title("📈 Prediksi Promosi Karyawan")

# Opsi input
opsi = st.radio("Pilih metode input:", ["Input Manual", "Unggah File CSV"])

if opsi == "Input Manual":
    with st.form("form_manual"):
        umur = st.number_input("🎂 Umur", min_value=18, max_value=60, value=30)
        jumlah_training = st.slider("🧠 Jumlah Pelatihan (2 tahun terakhir)", 0, 50, 2)
        rata_rata_skor_training = st.slider("📊 Rata-rata Skor Training", 0, 100, 50)
        masa_kerja = st.number_input("📅 Masa Kerja (tahun)", 0, 40, 5)
        rating_tahun_lalu = st.slider("⭐️ Rating Tahun Lalu", 1.0, 5.0, 3.0, step=0.1)
        KPI = st.radio("✅ KPI di atas 80%", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        penghargaan = st.radio("🏅 Pernah Menerima Penghargaan?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        departemen = st.selectbox("🏢 Departemen", [
            "Operations", "Technology", "Sales & Marketing", "Procurement",
            "Analytics", "Finance", "HR", "Legal", "R&D"
        ])
        wilayah = st.selectbox("📍 Wilayah", [f"wilayah_{i}" for i in range(1, 34)])
        pendidikan = st.selectbox("🎓 Pendidikan", ["Below Secondary", "Bachelor's", "Master's & above"])
        jenis_kelamin = st.radio("👤 Jenis Kelamin", ["m", "f"], format_func=lambda x: "Laki-laki" if x == "m" else "Perempuan")
        rekrutmen = st.selectbox("🧾 Sumber Rekrutmen", ["other", "sourcing", "referred"])
        submit_manual = st.form_submit_button("🔮 Prediksi")

    if submit_manual:
        data_manual = pd.DataFrame([{
            "jumlah_training": jumlah_training,
            "umur": umur,
            "rating_tahun_lalu": rating_tahun_lalu,
            "masa_kerja": masa_kerja,
            "KPI_>80%": KPI,
            "penghargaan": penghargaan,
            "rata_rata_skor_training": rata_rata_skor_training,
            "departemen": departemen,
            "wilayah": wilayah,
            "pendidikan": pendidikan,
            "jenis_kelamin": jenis_kelamin,
            "rekrutmen": rekrutmen
        }])
        df_pred = pd.get_dummies(data_manual)
        for col in fitur:
            if col not in df_pred:
                df_pred[col] = 0
        df_pred = df_pred[fitur]
        hasil = model.predict(df_pred)[0]
        proba = model.predict_proba(df_pred)[0][1]
        if hasil == 1:
            st.success(f"✅ Karyawan ini kemungkinan **AKAN DIPROMOSIKAN** (Probabilitas: {proba:.2%})")
        else:
            st.warning(f"❌ Karyawan ini kemungkinan **TIDAK dipromosikan** (Probabilitas: {proba:.2%})")
        st.dataframe(data_manual.T.rename(columns={0: "Data"}))

elif opsi == "Unggah File CSV":
    uploaded_file = st.file_uploader("Unggah file CSV dengan data karyawan", type="csv")
    if uploaded_file:
        data_csv = pd.read_csv(uploaded_file)
        st.write("📄 Data CSV yang diunggah:")
        st.dataframe(data_csv)

        # Preprocessing
        df_pred = pd.get_dummies(data_csv)
        for col in fitur:
            if col not in df_pred:
                df_pred[col] = 0
        df_pred = df_pred[fitur]

        # Prediksi masal
        hasil = model.predict(df_pred)
        proba = model.predict_proba(df_pred)[:, 1]

        # Gabungkan hasil ke data awal
        data_csv["Prediksi"] = ["Dipromosikan" if h == 1 else "Tidak" for h in hasil]
        data_csv["Probabilitas"] = [f"{p:.2%}" for p in proba]

        st.success("✅ Prediksi berhasil dilakukan!")
        st.dataframe(data_csv)
        st.download_button("⬇️ Unduh Hasil Prediksi", data=data_csv.to_csv(index=False), file_name="hasil_prediksi.csv", mime="text/csv")