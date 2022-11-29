import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv("data/train.csv")
y = df_train['price_range']

def minMax():
  st.write('Data Awal Sebelum di lakukan Preprocessing')
  st.dataframe(df_train)
  
  st.write('Data setelah dilakukan Preprocessing menggunakan Min-Max Scaler')
  scaler = MinMaxScaler()
  df_train_pre = scaler.fit_transform(df_train.drop(columns=["blue", "dual_sim", "four_g", "three_g" , "touch_screen", "wifi", "price_range"]))
  df_gabung = np.column_stack([df_train_pre, df_train[["blue", "dual_sim", "four_g", "three_g" , "touch_screen", "wifi"]]])
  st.write(df_gabung)
    
  # Save Scaled
  joblib.dump(df_gabung, 'model/df_train_pre.sav')
  joblib.dump(scaler,'model/df_scaled.sav')