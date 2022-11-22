import streamlit as st
import pandas as pd
from streamlit_extras.app_logo import add_logo

add_logo("http://placekitten.com/150/150")
st.markdown("# Home")
st.write("""
         Nama   : Ahmad Rosyihuddin\n
         NIM    : 200411100126\n
         Kelas  : Data Mining A\n
         """)

st.title("Dataset")
st.write('Dataset yang di gunakan merupakan Historis data dari Bitcoin dari tanggal 1 Februari 2018 sampai dengan 30 Oktober 2022')

# Load Dataset
dataset = pd.read_csv("Dataset/BitcoinDataset.csv")
st.write(dataset)

