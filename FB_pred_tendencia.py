# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:24:06 2020

@author: MarceloArce
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


datos = pd.read_csv('./FB.csv')
#CONVERTIMOS LOS DATOS EN UNA MATRIZ, SOLO LAS COLUMNAS 1 Y 0 FECHA Y PRECION DE ENTRADA

set_train = datos.iloc[:, 1:2].values

#Escalamos con sklearn(normalizar)

sc = MinMaxScaler(feature_range=(0, 1))

set_train_sc = sc.fit_transform(set_train)

#ESTRUCTURA CON 60 DIAS COMO PARAMETRO, POR CADA 60 DIAS ME LANZARA EL DATO DEL DIA SIGUIENTE

x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(set_train_sc[i-60: i, 0])
    y_train.append(set_train_sc[i,0])
#convertimos a matrices numpy  
x_train, y_train = np.array(x_train), np.array(y_train)

#AGREGAMOS UNA DIMENSION A LA MATRIZ NUMPY CON RESHAPE, EN EL XTRAIN DONDE ESTAN LOS INPUTS
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#Modelamos con Keras
regresor = Sequential()

#Agregamos capas imputs con LSTM, en este caso pondremos 50 celdas LSTM, un dropout del 20%
regresor.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1],1)))
regresor.add(Dropout(0.2))

#2da capa LSTM
regresor.add(LSTM(units=50, return_sequences=True))
regresor.add(Dropout(0.2))
#3ra capa LSTM
regresor.add(LSTM(units=50, return_sequences=True))
regresor.add(Dropout(0.2))
#4ta capa LSTM
regresor.add(LSTM(units=50))
regresor.add(Dropout(0.2))

#CREAMOS LA CAPA OUTPUT con Dense
regresor.add(Dense(units=1))
#USAREMOS el optimizador RSMPROP que es recomendable para redes neuronales recurrentes. Y ADAM
regresor.compile(optimizer='adam', loss='mean_squared_error')
#Con esto ya se tiene compilado, ahora pasaremos a entrenar el modelo
regresor.fit(x_train, y_train, epochs=100, batch_size=32)

#Trabajamos con el test que es el primer mes del siguiente año

datos_test = pd.read_csv('./FB_test.csv')
set_test = datos_test.iloc[:, 1:2].values
#Hacemos la concatenación de los datos train con los test con un axis 0 porque deseamos trabajar con la columna(0=columna 1=fila)
datos_total = pd.concat((datos['Open'], datos_test['Open']),axis=0)
inputs = datos_total[len(datos_total) - len(datos_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

#Le damos la dimension extra
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60: i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#hacemos la predicción
prediccion_precio = regresor.predict(x_test)
prediccion_precio = sc.inverse_transform(prediccion_precio)

#Visualizamos resultados
plt.plot(set_test, color='red', label='Precio real')
plt.plot(prediccion_precio, color='green', label= 'Precio predecido')
plt.title('Prediccion para Facebook en la bolsa para el año 2020')
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.legend()
plt.show()


