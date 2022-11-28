import streamlit as st
import pandas as pd
from streamlit_extras.app_logo import add_logo

add_logo("http://placekitten.com/150/150")
st.snow()
st.markdown("# Home")
st.text("""
         Nama   : Ahmad Rosyihuddin
         NIM    : 200411100126
         Kelas  : Data Mining A
         """)
st.write("""
         Email  : arosyihuddin6@gmail.com\n
         Github : [Github Repositori](https://github.com/a-rosyihuddin/streamlit-datamining)
         """)

st.markdown("# Klasisfikasi Range Harga Berdasarkan Spesifikasi Handphone")
st.text(""" Data yang di gunakan untuk klasifikasi range harga memiliki 17 fitur, 
dan dari 17 fitur ini sudah termasuk dengan label, yang value nya berupa
index dari 0-3 dengan penjelasan sebagai berikut:
  > 0 - Low Cost
  > 1 - Medium Cost
  > 2 - High Cost
  > 3 - Very High Cost """)


# Load Dataset

df_test = pd.read_csv("data/test.csv")



