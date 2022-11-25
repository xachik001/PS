import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import json

model1 = pickle.load(open('model1.pkl', 'rb')) #чтение модели
model2 = pickle.load(open('model2.pkl', 'rb')) #чтение модели


path = "data/ebw_data.csv" #загрузка датасета
df = pd.read_csv(path, sep = ",", encoding = "utf-8") #чтение данных
#convert all columns to float
df = df.astype(float)

st.title('Прогнозирование размеров сварного шва при электронно-лучевой сварке тонкостенных конструкций аэрокосмического назначения')

st.write('Введите данные для предсказания')

st.write(f"Значения величины сварочного тока в диапазоне от {df['IW'].min()} до {df['IW'].max()}")

#step=0.1, all values as float
IW = st.number_input('IW ', min_value=df['IW'].min(), max_value=np.float(df['IW'].max()), step=0.1, value=df['IW'].min())



st.write(f"Значения тока фокусировки электронного пучка в диапазоне от {df['IF'].min()} до {df['IF'].max()}")
IF = st.number_input('IF', min_value=df['IF'].min(), max_value=df['IF'].max(), step=0.1, value=df['IF'].min())

st.write(f"Значения скорости сварки в диапазоне от {df['VW'].min()} до {df['VW'].max()}")
VW = st.number_input('VW', min_value=df['VW'].min(), max_value=df['VW'].max(), step=0.1, value=df['VW'].min())

st.write(f"Значения расстояния от поверхности образцов до электронно-оптической системы в диапазоне от {df['FP'].min()} до {df['FP'].max()}")
FP = st.number_input('FP', min_value=df['FP'].min(), max_value=df['FP'].max(), step=0.1, value=df['FP'].min())

data = {'IW': IW, 'IF': IF, 'VW': VW, 'FP': FP} #создание словаря с данными

prediction1 = model1.predict(pd.DataFrame(data, index=[0])) #предсказание
prediction2 = model2.predict(pd.DataFrame(data, index=[0]))

# кнопка
# if st.button('Предсказать'):
st.write('Прогнозз глубины сварного шва: ', round(prediction1[0],2)) #вывод предсказаний
st.write('Прогнозз ширины сварного шва: ', round(prediction2[0],2))

