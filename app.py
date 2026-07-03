import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline

# 1. Konfigurasi Dasar Tampilan Halaman Web
st.set_page_config(page_title="Analisis Sentimen Kebijakan TKA", layout="wide")

# 2. Fungsi Cache Resource agar Model Tidak Dimuat Berulang Kali Setiap Klik
@st.cache_resource
def muat_seluruh_model():
    # Hapus tulisan 'model/' di depannya
    svm = joblib.load('svm_model.pkl')
    nb = joblib.load('nb_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    
    indobert = pipeline("sentiment-analysis", model="indobenchmark/indobert-base-p2")
    return svm, nb, tfidf, indobert

try:
    svm_model, nb_model, tfidf, indobert_pipeline = muat_seluruh_model()
except Exception as e:
    st.error(f"Gagal memuat komponen model. Pastikan struktur direktori folder 'model/' sudah benar. Deskripsi Error: {e}")

# 3. Layout Komponen Judul Utama
st.title("Sistem Klasifikasi Sentimen Masyarakat terhadap Kebijakan Tes Kemampuan Akademik (TKA)")
st.caption("Aplikasi Perbandingan Kinerja Algoritma Machine Learning Klasik dengan Model Arsitektur Transformer")
st.markdown("---")

# 4. Navigasi Menu Antarmuka Utama Menggunakan Komponen Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Teks Baru", "📊 Hasil Evaluasi Komparatif", "❌ Analisis Kesalahan (Error Analysis)"])

# TAB 1: PENGUJIAN MODEL SECARA REAL-TIME
with tab1:
    st.header("Uji Coba Prediksi Sentimen Kalimat")
    st.write("Silakan masukkan teks opini dari media sosial terkait pelaksanaan Tes Kemampuan Akademik (TKA):")
    
    user_input = st.text_area("Input Teks Opini:", height=130, placeholder="Tulis opini di sini...")
    
    if st.button("Jalankan Prediksi Klasifikasi"):
        if user_input.strip() != "":
            # Alur Pemrosesan Model Klasik (Membutuhkan Transformasi Vektor TF-IDF)
            # Opsional: Jika performa turun, Anda dapat menambahkan fungsi tokenisasi/cleansing ringan sebelum proses transform.
            text_vector = tfidf.transform([user_input])
            res_svm = svm_model.predict(text_vector)[0]
            res_nb = nb_model.predict(text_vector)[0]
            
            # Alur Pemrosesan Model Transformer IndoBERT (Teks Asli Masuk Langsung)
            raw_res_transformer = indobert_pipeline(user_input[:512])[0]
            res_trans = 'Positif' if raw_res_transformer['label'] == 'LABEL_1' else 'Negatif'
            
            # Tampilan Output Perbandingan Menggunakan Kolom Matriks Visual
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Prediksi Model SVM", value=res_svm)
            with col2:
                st.metric(label="Prediksi Model Naive Bayes", value=res_nb)
            with col3:
                st.metric(label="Prediksi Model IndoBERT", value=res_trans)
        else:
            st.warning("Kotak input teks tidak boleh kosong. Harap isi kalimat terlebih dahulu!")

# TAB 2: TABEL PERBANDINGAN HASIL EVALUASI METRIK
with tab2:
    st.header("Tabel Perbandingan Hasil Pengujian Model")
    st.write("Berikut merupakan rangkuman nilai metrik evaluasi yang diperoleh dari proses komparasi data uji (Test Set):")
    
    try:
        df_evaluasi_hasil = pd.read_csv("hasil_evaluasi.csv")
        st.dataframe(df_evaluasi_hasil, use_container_width=True)
        
        st.markdown("""
        ### Ringkasan Hasil Analisis:
        - **Model Performa Terbaik:** Tinjau tingkat capaian skor *Accuracy* dan *F1-score* tertinggi pada matriks tabel di atas.
        - **Karakteristik Model:** Model berbasis Transformer cenderung memiliki pemahaman semantik kontekstual kalimat yang lebih mendalam dibandingkan algoritma klasifikasi statistik konvensional.
        """)
    except FileNotFoundError:
        st.warning("Berkas data 'hasil_evaluasi.csv' tidak dapat ditemukan di dalam direktori folder 'model/'.")

# TAB 3: ERROR ANALYSIS SESUAI KETENTUAN LAPORAN
with tab3:
    st.header("Error Analysis Pelabelan Model")
    st.write("Menampilkan daftar 10 data sampel dari pengujian kalimat yang salah diprediksi beserta poin jabaran analisis penyebab kesalahan:")
    
    try:
        df_error_hasil = pd.read_csv("error_analysis.csv")
        st.table(df_error_hasil)
    except FileNotFoundError:
        st.warning("Berkas data 'error_analysis.csv' tidak dapat ditemukan di dalam direktori folder 'model/'.")
