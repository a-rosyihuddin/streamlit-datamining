import streamlit as st
import joblib
import numpy as np

price_range = {0 : 'Low Cost', 1 : 'Medium Cost', 2 : 'High Cost', 3 : 'Very High Cost'}


def knn(data_baru):
  model = joblib.load('model/knn_model.sav')
  hasil_predict = model.predict(data_baru)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {price_range[hasil_predict[0]]}')
