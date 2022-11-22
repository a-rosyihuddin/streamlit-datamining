import streamlit as st
st.title("Aplikasi Data Mining")

side_bar = st.sidebar.selectbox(
  'Pilih Dataset',
  ('Bungan IRIS', 'Kanker Payudara', 'Digit Angka')
)
