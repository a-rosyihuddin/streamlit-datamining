import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


df_train = pd.read_csv("data/train.csv")
x = df_train.drop(columns=["price_range"])
y = df_train['price_range']

def knn():
  # Preprocessing Min-Max Scaler
  df_train_pre = joblib.load('model/df_train_pre.sav')
  x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 1)
  scores = {}
  for i in range(1, 50+1):
      KN = KNeighborsClassifier(n_neighbors = i)
      KN.fit(x_train, y_train)
      y_pred = KN.predict(x_test)
      scores[i] = accuracy_score(y_test, y_pred)
      
  best_k = max(scores, key=scores.get)
  st.caption("Splitting Data yang digunakan merupakan 70:30, 30\% untuk data test dan 70\% untuk data train\nIterasi K di lakukan sebanyak 50 Kali")
  st.success(f"K Terbaik : {best_k} berada di Index : {best_k-1}, Akurasi Yang di Hasilkan : {max(scores.values())* 100}%")
  st.write(df_train_pre)
  
  # Create Chart 
  st.write('Dari proses pemodelan yang telah di lakukan menghasilkan grafik sebagai berikut')
  accuration_k = np.array(list(scores.values()))
  chart_data = pd.DataFrame(accuration_k, columns=['Score Akurasi'])
  st.line_chart(chart_data)
  
  # Save Model
  knn = KNeighborsClassifier(n_neighbors=best_k)
  knn.fit(x_train, y_train)
  joblib.dump(knn, 'model/knn_model.sav') # Menyimpan Model ke dalam folder model
  
  
  # Tanpa Preprocessing
  x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(x, y, test_size = 0.3, random_state = 1)
  scores_np = {}
  for i in range(1, 50+1):
      KN = KNeighborsClassifier(n_neighbors = i)
      KN.fit(x_train_np, y_train_np)
      y_pred_np = KN.predict(x_test_np)
      scores_np[i] = accuracy_score(y_test_np, y_pred_np)

      
  best_k_np = max(scores_np, key=scores_np.get)
  st.caption("Splitting Data yang digunakan merupakan 70:30, 30\% untuk data test dan 70\% untuk data train\nIterasi K di lakukan sebanyak 50 Kali")
  st.success(f"K Terbaik : {best_k_np} berada di Index : {best_k_np-1}, Akurasi Yang di Hasilkan : {max(scores_np.values())* 100}%")
  st.write(x)
  
  # Create Chart 
  st.write('Dari proses pemodelan yang telah di lakukan menghasilkan grafik sebagai berikut')
  accuration_k_np = np.array(list(scores_np.values()))
  chart_data = pd.DataFrame(accuration_k_np, columns=['Score Akurasi'])
  st.line_chart(chart_data)


def dcc():
  # Preprocessing Min-Max Scaler
  df_train_pre = joblib.load('model/df_train_pre.sav')
  x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 1)
  
  dcc = DecisionTreeClassifier()
  dcc.fit(x_train, y_train)
  # Save Model
  joblib.dump(dcc, 'model/dcc_model.sav') # Menyimpan Model ke dalam folder model
  
  y_pred = dcc.predict(x_test)
  akurasi = accuracy_score(y_test,y_pred)
  
  st.caption("Splitting Data yang digunakan merupakan 70:30, 30\% untuk data test dan 70\% untuk data train")
  st.success(f'Akurasi Yang di dapatkan adalah : {akurasi*100}%')
  st.write(df_train_pre)
  
  # Tanpa Preprocessing
  x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(x, y, test_size = 0.3, random_state = 1)
  
  dcc_np = DecisionTreeClassifier()
  dcc_np.fit(x_train_np, y_train_np)
  # Save Model
  joblib.dump(dcc_np, 'model/dcc_model.sav') # Menyimpan Model ke dalam folder model
  
  y_pred_np = dcc_np.predict(x_test_np)
  akurasi = accuracy_score(y_test_np,y_pred_np)
  
  st.caption("Splitting Data yang digunakan merupakan 70:30, 30\% untuk data test dan 70\% untuk data train")
  st.success(f'Akurasi Yang di dapatkan adalah : {akurasi*100}%')
  st.write(x)

def nb():
  # Preprocessing Min-Max Scaler
  df_train_pre = joblib.load('model/df_train_pre.sav')
  x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
  
  nb = GaussianNB()
  nb.fit(x_train, y_train)
  # Save Model
  joblib.dump(nb, 'model/nb_model.sav') # Menyimpan Model ke dalam folder model
  
  y_pred = nb.predict(x_test)
  akurasi = accuracy_score(y_test,y_pred)
  
  st.caption("Splitting Data yang digunakan merupakan 70:30, 30\% untuk data test dan 70\% untuk data train")
  st.success(f'Akurasi Yang di dapatkan adalah : {akurasi*100}%')
  st.write(df_train_pre)
  
  # Tanpa Preprocessing
  x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(x, y, test_size = 0.3, random_state = 0)
  
  nb_np = GaussianNB()
  nb_np.fit(x_train_np, y_train_np)
  # Save Model
  joblib.dump(nb_np, 'model/nb_model.sav') # Menyimpan Model ke dalam folder model
  
  y_pred_np = nb_np.predict(x_test_np)
  akurasi = accuracy_score(y_test_np,y_pred_np)
  
  st.caption("Splitting Data yang digunakan merupakan 70:30, 30\% untuk data test dan 70\% untuk data train")
  st.success(f'Akurasi Yang di dapatkan adalah : {akurasi*100}%')
  st.write(x)