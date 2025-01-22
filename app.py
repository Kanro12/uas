import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Fungsi untuk melatih model
def train_model():
    # Load dataset
    df = pd.read_csv("insurance1.csv")

    # Define X and y
    X = df[['age', 'sex', 'bmi', 'children', 'smoker']]
    y = df['charges']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'insurance_model.pkl')

    return model

# Fungsi untuk memuat model
def load_model():
    return joblib.load('insurance_model.pkl')

# Cek jika model sudah dilatih sebelumnya
if not os.path.exists('insurance_model.pkl'):
    st.write("Melatih model baru...")
    train_model()

# Load model
model = load_model()

# Streamlit App
st.title("Prediksi Biaya Asuransi")

# Menampilkan nama pembuat dan NIM di kanan atas
st.markdown(
    """
    <style>
    .stApp {
        padding-top: 50px;
    }
    .name {
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 16px;
        color: #333;
    }
    </style>
    <div class="name">
        
    </div>
    """, unsafe_allow_html=True)

# Input form
age = st.number_input("Umur", min_value=18, max_value=100, value=30, step=1)
sex = st.selectbox("Jenis Kelamin", options=["male", "female"])
bmi = st.number_input("Berat Badan (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Perokok", options=["yes", "no"])

# Konversi input ke format numerik
sex_encoded = 0 if sex == "male" else 1
smoker_encoded = 1 if smoker == "yes" else 0

# Prediksi
if st.button("Prediksi Biaya Asuransi"):
    try:
        # Buat array input untuk prediksi
        input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded]])
        predicted_charges = model.predict(input_data)

        # Tampilkan hasil prediksi
        st.success(f"Prediksi Biaya Asuransi: ${predicted_charges[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")
