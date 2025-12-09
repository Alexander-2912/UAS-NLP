# app.py
import re
import string
import joblib
import numpy as np
import streamlit as st

# NLTK & Sastrawi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def download_nltk():
    # punkt
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

download_nltk()

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words('indonesian'))

negations = {"tidak", "bukan", "jangan", "tak", "nggak", "gak", "ga"}
stop_words = stop_words - negations

custom_add = {
    "yg", "aja", "dong", "nih", "sih", "banget", "bgt",
    "rt", "amp", "url", "hehe", "wkwk", "wk", "haha",
    "pls", "tolong", "kalo", "kl", "gitu", "kayak", "dong"
}
stop_words = stop_words | custom_add

stop_words_stemmed = {stemmer.stem(w) for w in stop_words}

def clean_text(text: str) -> str:
    # 1) Lowercase
    text = text.lower()
    # 2) Hapus URL, mention, hashtag
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    # 3) Hapus angka & tanda baca
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 4) Rapikan spasi
    text = re.sub(r'\s+', ' ', text).strip()
    # 5) Tokenisasi
    tokens = word_tokenize(text)
    # 6) Stemming
    tokens = [stemmer.stem(w) for w in tokens]
    # 7) Stopword removal
    tokens = [w for w in tokens if w not in stop_words_stemmed]
    # 8) Gabung lagi
    return " ".join(tokens)

@st.cache_resource
def load_models():
    pipe_bow = joblib.load("pipeline_bow_lgbm_7030.joblib")
    pipe_tfidf = joblib.load("pipeline_tfidf_lgbm_7030.joblib")
    return pipe_bow, pipe_tfidf

pipe_bow, pipe_tfidf = load_models()






st.set_page_config(
    page_title="Indonesian Hate Speech Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("Indonesian Hate Speech Classification")
st.write(
    "Aplikasi ini memprediksi apakah sebuah kalimat termasuk **hate speech** "
    "atau **neutral** menggunakan model **LightGBM**.\n\n"
    "Kamu bisa memilih antara model **BoW** atau **TF-IDF**."
)

# Pilihan model
model_choice = st.radio(
    "Pilih model yang ingin digunakan:",
    ("LightGBM + BoW", "LightGBM + TF-IDF")
)

# Input teks
user_text = st.text_area(
    "Masukkan kalimat (Bahasa Indonesia):",
    height=120,
    placeholder="Contoh: 'Dasar bodoh, kamu nggak pantas di sini!'"
)

# Tombol prediksi
if st.button("🔍 Prediksi"):
    if not user_text.strip():
        st.warning("Silakan masukkan kalimat terlebih dahulu.")
    else:
        # Preprocess (harus sama dengan saat training)
        cleaned = clean_text(user_text)

        # Pilih pipeline
        if model_choice == "LightGBM + BoW":
            model = pipe_bow
        else:
            model = pipe_tfidf

        # Prediksi
        pred_label = model.predict([cleaned])[0]

        # Probabilitas (jika tersedia)
        proba = None
        try:
            proba = model.predict_proba([cleaned])[0]
            classes = model.classes_
        except Exception:
            proba = None
            classes = None

        # Tampilkan hasil
        st.subheader("Hasil Prediksi")

        if str(pred_label).lower() == "hate":
            st.error(f"Prediksi: **HATE SPEECH** (`{pred_label}`)")
        else:
            st.success(f"Prediksi: **NEUTRAL** (`{pred_label}`)")

        st.markdown("**Teks asli:**")
        st.write(user_text)

        st.markdown("**Teks setelah preprocessing:**")
        st.code(cleaned)

        # Tampilkan probabilitas
        if proba is not None and classes is not None:
            st.markdown("**Probabilitas kelas:**")
            for cls, p in zip(classes, proba):
                st.write(f"- `{cls}` : `{p:.4f}`")