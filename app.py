import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Judul aplikasi
st.title("🫀 Aplikasi Prediksi Penyakit Jantung")
st.write("Masukkan data pasien untuk memprediksi risiko penyakit jantung.")

# Load dataset (ganti path sesuai lokasi file heart.csv di komputer kamu)
df = pd.read_csv("heart.csv")

# Pisahkan fitur dan target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(solver='liblinear', C=1.623776739188721)
model.fit(X_train_scaled, y_train)

# Sidebar input form
st.sidebar.header("📝 Masukkan Data Pasien")

def user_input_features():
    age = st.sidebar.slider("Usia", 20, 100, 50)
    sex = st.sidebar.selectbox("Jenis Kelamin (0=Wanita, 1=Pria)", [0, 1])
    cp = st.sidebar.selectbox("Jenis Nyeri Dada (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Tekanan Darah Saat Istirahat (mmHg)", 80, 200, 120)
    chol = st.sidebar.slider("Kolesterol Serum (mg/dl)", 100, 600, 250)
    fbs = st.sidebar.selectbox("Gula Darah Puasa > 120 mg/dl? (0=Tidak, 1=Ya)", [0, 1])
    restecg = st.sidebar.selectbox("Hasil EKG Istirahat (0-2)", [0, 1, 2])
    thalach = st.sidebar.slider("Detak Jantung Maksimum", 70, 220, 150)
    exang = st.sidebar.selectbox("Nyeri Dada karena Olahraga? (0=Tidak, 1=Ya)", [0, 1])
    oldpeak = st.sidebar.slider("Depresi ST karena Olahraga relatif terhadap istirahat", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope segmen ST saat olahraga (0-2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Jumlah pembuluh darah besar yang diwarnai fluoroskopi (0-4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Tampilkan input pengguna
st.subheader("📋 Data Pasien")
st.write(input_df)

# Scaling input
input_scaled = scaler.transform(input_df)

# Prediksi
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Hasil prediksi
st.subheader("🔮 Hasil Prediksi")
if prediction[0] == 1:
    st.error("⚠️ Pasien kemungkinan besar menderita penyakit jantung.")
else:
    st.success("✅ Pasien tidak menunjukkan gejala penyakit jantung.")

# Probabilitas
st.write(f"Probabilitas: {prediction_proba[0][prediction[0]]:.2%}")

# Akurasi model
st.sidebar.subheader("📊 Akurasi Model")
st.sidebar.write(f"{accuracy_score(y_test, model.predict(X_test_scaled)) * 100:.2f}%")