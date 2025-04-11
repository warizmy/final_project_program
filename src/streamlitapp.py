import streamlit as st
import joblib
import pandas as pd

# Load model dan sclaer
model = joblib.load('../models/best_rf_model.joblib')
scaler = joblib.load('../models/standard_scaler.joblib')

st.set_page_config(
    page_title="Prediksi Ketepatan Lulus Mahasiswa", 
    page_icon="../public/assets/favicon.ico" 
)

def preprocess_input(input_df):
    cols_to_drop = ['NIM', 'Nama', 'IPS 7', 'IPS 5', 'Total SKS']
    processed_df = input_df.drop(columns=cols_to_drop, errors='ignore')
    
    numeric_cols = ['IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 6', 'IPS 8', 'IPK Rata rata']
    processed_df[numeric_cols] = scaler.transform(processed_df[numeric_cols])
    
    return processed_df

def main():
    st.title('Prediksi Ketepatan Lulus Mahasiswa 🎓')
    
    with st.form("input_form"):
        st.subheader("Data Mahasiswa")
        nim = st.text_input('NIM')  
        nama = st.text_input('Nama') 
        
        st.subheader("Indeks Prestasi Semester")
        ips1 = st.number_input('IPS 1 (Indeks Prestasi Semester 1)', min_value=0.0, max_value=4.0, step=0.01)
        ips2 = st.number_input('IPS 2 (Indeks Prestasi Semester 2)', min_value=0.0, max_value=4.0, step=0.01)
        ips3 = st.number_input('IPS 3 (Indeks Prestasi Semester 3)', min_value=0.0, max_value=4.0, step=0.01)
        ips4 = st.number_input('IPS 4 (Indeks Prestasi Semester 4)', min_value=0.0, max_value=4.0, step=0.01)
        ips5 = st.number_input('IPS 5 (Indeks Prestasi Semester 5)', min_value=0.0, max_value=4.0, step=0.01)  
        ips6 = st.number_input('IPS 6 (Indeks Prestasi Semester 6)', min_value=0.0, max_value=4.0, step=0.01)
        ips7 = st.number_input('IPS 7 (Indeks Prestasi Semester 7)', min_value=0.0, max_value=4.0, step=0.01) 
        ips8 = st.number_input('IPS 8 (Indeks Prestasi Semester 8)', min_value=0.0, max_value=4.0, step=0.01)
        
        st.subheader("Lainnya")
        ipk = st.number_input('IPK Rata-rata', min_value=0.0, max_value=4.0, step=0.01)
        total_sks = st.number_input('Total SKS', min_value=0)
        
        submit_button = st.form_submit_button("Prediksi 🚀")

    if submit_button:
        input_data = pd.DataFrame([[
            nim, nama, ips1, ips2, ips3, ips4, ips5, ips6, ips7, ips8, total_sks, ipk
        ]], columns=[
            'NIM', 'Nama', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 
            'IPS 6', 'IPS 7', 'IPS 8', 'Total SKS', 'IPK Rata rata'
        ])
        
        processed_data = preprocess_input(input_data)
        
        prediction = model.predict(processed_data)
        
        result = "Lulus Tepat Waktu ✅" if prediction[0] == 1 else "Tidak Lulus Tepat Waktu ❌"
        st.success(f"Hasil Prediksi: {result}")

if __name__ == '__main__':
    main()