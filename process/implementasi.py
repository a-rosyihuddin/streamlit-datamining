import streamlit as st
import joblib
from sklearn.metrics import accuracy_score

price_range = {0 : 'Low Cost', 1 : 'Medium Cost', 2 : 'High Cost', 3 : 'Very High Cost'}


def knn(data, data_scaler):
  model = joblib.load('model/knn_model.sav')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset tanpa di lakukan normalisasi Min-Max Scaler')
  y_pred = model.predict(data)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {price_range[y_pred[0]]}')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset dengan di lakukan normalisasi Min-Max Scaler')
  y_pred_scaler = model.predict(data_scaler)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {price_range[y_pred_scaler[0]]}')

def dcc(data, data_scaler):
  model = joblib.load('model/dcc_model.sav')
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset tanpa di lakukan normalisasi Min-Max Scaler')
  y_pred = model.predict(data)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {price_range[y_pred[0]]}')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset dengan di lakukan normalisasi Min-Max Scaler')
  y_pred_scaler = model.predict(data_scaler)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {price_range[y_pred_scaler[0]]}')
  
def nb(data, data_scaler):
  model = joblib.load('model/nb_model.sav')
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset tanpa di lakukan normalisasi Min-Max Scaler')
  y_pred = model.predict(data)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {price_range[y_pred[0]]}')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset dengan di lakukan normalisasi Min-Max Scaler')
  y_pred_scaler = model.predict(data_scaler)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {price_range[y_pred_scaler[0]]}')
  