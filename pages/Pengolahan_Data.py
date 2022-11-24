import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from streamlit_extras.app_logo import add_logo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def loading():
  with st.spinner('Tunggu Sebentar...'):
    time.sleep(1.5)

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

elif(selected == 'Preprocessing'):
  loading()
  st.write('Data dilakukan Preprocessing menggunakan Min-Max Scaler')
  scaled = MinMaxScaler()
  df_train_pre = scaled.fit_transform(df_train.drop(columns=["price_range"]))
  st.dataframe(df_train)
  

elif(selected == 'Modeling'):
  # Preprocessing Min-Max Scaler
  scaled = MinMaxScaler()
  df_train_pre = scaled.fit_transform(df_train.drop(columns=["price_range"]))
  x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.2, random_state = 0)
  loading()
  st.write('Modeling dengan menggunakan Dataset yang telah dilakukan preprocessing Min-Max Scaler')
  knn, dcc, nb = st.tabs(['K-Nerest Neighbor', 'Decission Tree', 'Naive Bayes'])
  with knn:
    scores = {}

    for i in range(1, 20+1):
        KN = KNeighborsClassifier(n_neighbors = i)
        KN.fit(x_train, y_train)
        y_pred = KN.predict(x_test)
        scores[i] = accuracy_score(y_test, y_pred)
        
    best_k = max(scores, key=scores.get)
    st.caption("Splitting Data yang digunakan merupakan 80:20, 20\% untuk data test dan 80\% untuk data train\nIterasi K di lakukan sebanyak 20 Kali")
    st.success(f"K Terbaik : {best_k} berada di Index : {best_k-1}, Akurasi Yang di Hasilkan : {max(scores.values())* 100}%")
    st.write(df_train_pre)
    
    # Create Chart 
    st.write('Dari proses pemodelan yang telah di lakukan menghasilkan grafik sebagai berikut')
    accuration_k = np.array(list(scores.values()))
    chart_data = pd.DataFrame(accuration_k, columns=['Score Akurasi'])
    st.line_chart(chart_data)
    
    # Save Model
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(x_train, y_train)
    dirname = os.path.dirname(__file__)   # Mndapatkan Path directori
    joblib.dump(model, f'{dirname}/../model/knn_model_pre.sav') # Menyimpan Model ke dalam folder model
    
    # Tanpa Preprocessing Min-Max Scaler
    st.write('Modeling Data dengan menggunakan Dataset yang tidak dilakukan preprocessing Min-Max Scaler')
    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(df_train, y, test_size = 0.2, random_state = 0)
    with knn:
      scores_np = {}

      for i in range(1, 20+1):
          KN = KNeighborsClassifier(n_neighbors = i)
          KN.fit(x_train_np, y_train_np)
          y_pred_np = KN.predict(x_test_np)
          scores_np[i] = accuracy_score(y_test_np, y_pred_np)

      best_k_np = max(scores_np, key=scores_np.get)
      st.success(f"K Terbaik : {best_k_np} berada di Index : {best_k_np-1}, Akurasi Yang di Hasilkan : {max(scores_np.values())* 100}%")
      st.write(df_train)

      # Create Chart 
      st.write('Dari proses pemodelan yang telah di lakukan menghasilkan grafik sebagai berikut')
      accuration_k_np = np.array(list(scores_np.values()))
      chart_data_np = pd.DataFrame(accuration_k_np, columns=['Score Akurasi'])
      st.line_chart(chart_data_np)

      # Save Model
      model_np = KNeighborsClassifier(n_neighbors=best_k_np)
      model_np.fit(x_train_np, y_train)# Nama File Penyimpanan
      dirname = os.path.dirname(__file__)   # Mndapatkan Path directori
      joblib.dump(model_np, f'{dirname}/../model/knn_model_np.sav')


elif(selected== 'Implementation'):
  loading()
  col1, col2 = st.columns(2)
  with col1:
    battery = st.number_input('Battery Power')
    bluetooth = st.number_input('Bluetoth')
    sim_card = st.number_input('Dual SIM')
    kamera_depan = st.number_input('Ukuran Kamera Depan(Mega Pikesel)')
    jaringan_4G = st.number_input('Jaringan 4G')
    int_memori= st.number_input('Internal Memori')
    berat_hp= st.number_input('Berat Handphone')
    kamera_belakang= st.number_input('Kemera Belakang (Mega Piksel)')
  with col2:
    tinggi_hp= st.number_input('Tinggi Handphone')
    lebar_hp= st.number_input('Lebar Handphone')
    ram= st.number_input('Ukuran RAM')
    tinngi_layar= st.number_input('Tinggi Layar')
    lebar_layar= st.number_input('Lebar Layar')
    jaringan_3G= st.number_input('Jaringan 3G')
    touchscreen= st.number_input('Touchscreen')
    wifi= st.number_input('WIFI')