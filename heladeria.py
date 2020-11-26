# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 18:37:01 2020

@author: Sys10
"""


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datos = pd.read_csv("C:/Users/Sys10/Desktop/modelo predictivo/heladeria.csv")

sns.regplot(datos['Temperature'], datos['Revenue'])

x_train = datos['Temperature']
y_train = datos['Revenue'] 

#creando model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape =[1]))
model.summary()

#compilacion con optimizador Adam
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss= 'mean_squared_error')


#Entrenamiento
epochs_hist = model.fit(x_train, y_train, epochs =1000)

keys = epochs_hist.history.keys()

#Grafico
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso en la perdida')
plt.xlabel('epochs')
plt.ylabel('training_loss')
plt.legend(['training_loss'])

weights = model.get_weights()

#Prediccion

Temp = 10
Revenue = model.predict([Temp])
print('La ganancia esperada si la Temperatura es igual a ', Temp, 'grados celsius es de: ', Revenue)

#graficando

plt.scatter(x_train, y_train, color='gray')
plt.plot(x_train, model.predict(x_train), color='red')
plt.ylabel('Ganancia en dolares')
plt.xlabel('Temperatura en Celsius')
plt.title('Temperatura vs ganancias')



