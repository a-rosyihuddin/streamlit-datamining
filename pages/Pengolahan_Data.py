import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from streamlit_option_menu import option_menu
from streamlit_extras.app_logo import add_logo

from process import implementasi 
from process import model
from process import preprocessing

def loading():
  with st.spinner('Tunggu Sebentar...'):
    time.sleep(0.3)

add_logo("http://placekitten.com/150/150")
st.markdown("# Pengolahan Data")
selected = option_menu(
    menu_title  = "Data Mining ",
    options     = ["Dataset","Preprocessing","Modeling","Implementation"],
    icons       = ["data","Process","model","implemen","Test"],
    orientation = "horizontal",
)

df_train = pd.read_csv("data/train.csv")
y = df_train['price_range']


if(selected == "Dataset"):
  loading()
  st.success(f"Jumlah Data : {df_train.shape[0]} Data, dan Jumlah Fitur : {df_train.shape[1]} Fitur")
  dataframe, keterangan = st.tabs(['Datset', 'Keterangan'])
  with dataframe:
    st.write(df_train)

  with keterangan:
    st.text("""
             Column:
             - Battery power: Berapa Banyak Power Dari Baterai
             - Blue: Apakah Batrey nya memiliki Bluetooth atau TIDAK
             - Dual_Sim: Apakah Mendukung Dual SIM atau TIDAK
             - fc: Ukuran Pixel Dari Kamera Depan
             - four_g: Apakah Sudah support jaringan 4G atau TIDAK
             - int_memory: Internal Memory berapa GB
             - mobile_wt: Berat Handphone
             - pc: Ukuran Pixel Dari Kamera Belakang/Primary
             - px_height: Pixel Resolution Height
             - px_width: Pixel Resolution Width
             - ram: Ukuran RAM
             - sc_h: Screen Height of mobile in cm
             - sc_w: Screen Width of mobile in cm
             - three_g: Apakah Jaringan nya support 3G
             - touch_screen: Layarnya Bisa di sentuh Atau tidak
             - wifi: Memiliki Jaringan WIFI atau Tidak
             - Price range: label dari kisaran harga
             
             Index
             Output Dari Dataset ini merupakan sebuah index yaitu : 0,1,2,3, 
             dimana dari 4 index ini di kategorikan sebagai berikut
             > 0 - Low Cost
             > 1 - Medium Cost
             > 2 - High Cost
             > 3 - Very High Cost
           """)

########################################## Preprocessing #####################################################
elif(selected == 'Preprocessing'):
  loading()
  preprocessing.minMax()
  
########################################### Modeling ##########################################################

elif(selected == 'Modeling'):
  loading()
  knn, dcc, nb = st.tabs(['K-Nerest Neighbor', 'Decission Tree', 'Naive Bayes'])
  with knn:
    model.knn()
  
  with dcc:
    model.dcc()
  
  with nb:
    model.nb()

####################################### Implementasi ###########################################################
elif(selected== 'Implementation'):
  col1, col2 = st.columns(2)
  with col1:
    battery = st.number_input('Battery Power (mAh', min_value=0, value=1021)
    bluetooth = st.selectbox('Bluetoth', ('Tidak','Ada'), index=1)
    sim_card = st.selectbox('Dual SIM', ('Tidak','Bisa'), index=1)
    kamera_depan = st.number_input('Ukuran Kamera Depan (Mega Pikesel)', min_value=0, value=0)
    jaringan_4G = st.selectbox('Jaringan 4G', ('Tidak', 'Support'), index=1)
    int_memori= st.number_input('Internal (GB)', min_value=0, value=53)
    berat_hp= st.number_input('Berat Handphone (g)', min_value=0, value=136)
    kamera_belakang= st.number_input('Kemera Belakang (Mega Piksel)', min_value=0, value=6)

  with col2:
    tinggi_hp= st.number_input('Tinggi Handphone (mm)', min_value=0, value=905)
    lebar_hp= st.number_input('Lebar Handphone (mm)', min_value=0, value=1988)
    ram= st.number_input('Ukuran RAM (GB)', min_value=0, value=2631)
    tinggi_layar= st.number_input('Tinggi Layar (pixel)', min_value=0, value=17)
    lebar_layar= st.number_input('Lebar Layar (pixel)', min_value=0, value=3)
    jaringan_3G= st.selectbox('Jaringan 3G', ('Tidak', 'Support'), index=1)
    touchscreen= st.selectbox('Touchscreen', ('Tidak','Iya'), index=1)
    wifi= st.selectbox('WIFI', ('Tidak', 'Ada'), index=0)
  ind_bluetooth = ('Tidak', 'Ada').index(bluetooth) 
  ind_sim_card = ('Tidak', 'Bisa').index(sim_card) 
  ind_jaringan_4G = ('Tidak', 'Support').index(jaringan_4G) 
  ind_jaringan_3G = ('Tidak', 'Support').index(jaringan_3G) 
  ind_touchscreen = ('Tidak', 'Iya').index(touchscreen) 
  ind_wifi = ('Tidak', 'Ada').index(wifi)
  data = np.array([[battery, ind_bluetooth, ind_sim_card, kamera_depan, ind_jaringan_4G, int_memori, berat_hp, kamera_belakang,
               tinggi_hp, lebar_hp, ram, tinggi_layar, lebar_layar, ind_jaringan_3G, ind_touchscreen, ind_wifi]])
  
  scaler = joblib.load('model/df_scaled.sav')
  data_scaler = scaler.fit_transform(data)
  
  knn, dcc, nb = st.tabs(['K-Nerest Neighbor', 'Decission Tree', 'Naive Bayes'])
  with knn:
    implementasi.knn(data, data_scaler)
    
  with dcc:
    implementasi.dcc(data, data_scaler)
  
  with nb:
    implementasi.nb(data, data_scaler)