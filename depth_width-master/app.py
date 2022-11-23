import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import json

model1 = pickle.load(open('model1.pkl', 'rb')) #чтение модели
model2 = pickle.load(open('model2.pkl', 'rb')) #чтение модели

# model1 = pickle.load('model1.pkl', 'rb') #чтение модели
# model2 = pickle.load('model2.pkl', 'rb') #чтение модели
path = "data/ebw_data.csv"
df = pd.read_csv(path, sep = ",", encoding = "utf-8") #чтение данных

st.title('Предсказание размеров')

st.write('Введите данные для предсказания')

st.write(f"Значения в диапазоне от {df['IW'].min()} до {df['IW'].max()}")
IW = st.number_input('IW', min_value=df['IW'].min(), max_value=df['IW'].max(), value=np.int64(df['IW'].mean()), step=np.int64(1))


st.write(f"Значения в диапазоне от {df['IF'].min()} до {df['IF'].max()}")
IF = st.number_input('IF', min_value=df['IF'].min(), max_value=df['IF'].max(), value=np.int64(df['IF'].mean()), step=np.int64(1))

st.write(f"Значения в диапазоне от {df['VW'].min()} до {df['VW'].max()}")
VW = st.number_input('VW', min_value=np.int64(df['VW'].min()), max_value=np.int64(df['VW'].max()), value=np.int64(df['VW'].mean()), step=np.int64(1))

st.write(f"Значения в диапазоне от {df['FP'].min()} до {df['FP'].max()}")
FP = st.number_input('FP', min_value=np.int64(df['FP'].min()), max_value=np.int64(df['FP'].max()), value=np.int64(df['FP'].mean()), step=np.int64(1))

data = {'IW': IW, 'IF': IF, 'VW': VW, 'FP': FP} #создание словаря с данными

prediction1 = model1.predict(pd.DataFrame(data, index=[0])) #предсказание
prediction2 = model2.predict(pd.DataFrame(data, index=[0]))

# prediction button
# if st.button('Предсказать'):
st.write('прогнозирование глубины(depth): ', round(prediction1[0],2))
st.write('прогнозирование ширины(width): ', round(prediction2[0],2))

