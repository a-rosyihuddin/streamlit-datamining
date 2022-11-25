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

st.markdown("# Apa Itu Data Mining")
st.caption("Data mining adalah proses pengumpulan dan pengolahan data yang bertujuan untuk mengekstrak informasi penting pada data. Proses pengumpulan dan ekstraksi informasi tersebut dapat dilakukan menggunakan perangkat lunak dengan bantuan perhitungan statistika, matematika, ataupun teknologi Artificial Intelligence (AI). Data mining sering disebut juga Knowledge Discovery in Database (KDD).")


# Load Dataset

df_test = pd.read_csv("data/test.csv")



