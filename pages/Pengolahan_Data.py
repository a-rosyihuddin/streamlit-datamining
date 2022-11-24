import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from streamlit_extras.app_logo import add_logo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

add_logo("http://placekitten.com/150/150")
st.markdown("# Pengolahan Data")
selected = option_menu(
    menu_title  = "Data Mining ",
    options     = ["Dataset","Preprocessing","Modeling","Implementation","Testing"],
    icons       = ["data","Process","model","implemen","Test"],
    orientation = "horizontal",
)

df_train = pd.read_csv("data/train.csv")
y = df_train['price_range']

if(selected == "Dataset"):
  st.success(f"Jumlah Data : {df_train.shape[0]} Data, dan Jumlah Fitur : {df_train.shape[1]} Fitur")
  dataframe, keterangan = st.tabs(['Datset', 'Keterangan'])
  with dataframe:
    st.write(df_train)

  with keterangan:
    st.text("""
             Output Dari Dataset ini merupakan categori : 0,1,2,3
             Column:
             * Battery power: Berapa Banyak Power Dari Baterai
             * Blue: Apakah Batrey nya memiliki Bluetooth atau TIDAK
             * Dual_Sim: Apakah Mendukung Dual SIM atau TIDAK
             * fc: Ukuran Pixel Dari Kamera Depan
             * four_g: Apakah Sudah support jaringan 4G atau TIDAK
             * int_memory: Internal Memory berapa GB
             * mobile_wt: Weight of mobile phone
             * pc: Ukuran Pixel Dari Kamera Depan/Primary
             * px_height: Pixel Resolution Height
             * px_width: Pixel Resolution Width
             * ram: Ukuran RAM
             * sc_h: Screen Height of mobile in cm
             * sc_w: Screen Width of mobile in cm
             * three_g: Apakah Jaringan nya support 3G
             * touch_screen: Layarnya Bisa di sentuh Atau tidak
             * wifi: Memiliki Jaringan WIFI atau Tidak
             * Price range: label dari kisaran harga
             Index
             * 0 - Low Cost
             * 1 - Medium Cost
             * 2 - High Cost
             * 3 - Very High Cost
           """)

elif(selected == 'Preprocessing'):
  st.write('Data dilakukan Preprocessing menggunakan Min-Max Scaler')
  scaled = MinMaxScaler()
  df_train = scaled.fit_transform(df_train.drop(columns=["price_range"]))
  st.dataframe(df_train)
  

elif(selected == 'Modeling'):
  x_train, x_test, y_train, y_test = train_test_split(df_train, y, test_size = 0.2, random_state = 0)
  knn, dcc, nb = st.tabs(['K-Nerest Neighbor', 'Decission Tree', 'Naive Bayes'])
  with knn:
    scores = {}
    scores_list = []

    for i in range(1, 20+1):
        KN = KNeighborsClassifier(n_neighbors = i)
        KN.fit(x_train, y_train)
        y_pred = KN.predict(x_test)
        scores[i] = accuracy_score(y_test, y_pred)
        scores_list.append(accuracy_score(y_test, y_pred))
    # best_k = 
    st.success(f"K Terbaik : {max(scores, key=scores.get)}, Akurasi Yang di Hasilkan : {max(scores.values())* 100}%")
    st.caption("Splitting Data yaitu 80:20, 20\% untuk data test dan 80\% untuk data train\nIterasi K di lakukan sebanyak 20 Kali")
    
    # Create Chart 
    accuration_k = np.array(list(scores.values()))
    chart_data = pd.DataFrame(accuration_k, columns=['Score Akurasi'])
    st.line_chart(chart_data)
