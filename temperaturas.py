# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:39:15 2020

@author: Sys10
"""


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datos = pd.read_csv("C:/Users/Sys10/Desktop/modelo predictivo/temperaturas.csv")

#Viendo datos con sns
sns.regplot(datos['Celsius'],datos['Fahrenheit'])


#Definiendo datos de entrenamiento
x_train = datos['Celsius']
y_train = datos['Fahrenheit']

#Creando el modelo, con tensorflow, keras para armar nuestro modelo, Sequential significa que el modelo sera secuencial (capa por capa)
model = tf.keras.Sequential()     

#creamos un input para los datos de entrada, es decir la primera capa.


model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#model.summary()

#Compilado y optimzacion que ayuda a crear las funciones de perdida y optimizar el sesgo del modelo

model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')

#Entrenamos con epoch para que recorra todos los datos, en este caso 100 veces
epochs_hist = model.fit(x_train, y_train, epochs = 100)

#Evaluamos
epochs_hist.history.keys()
#Graficamos en base a la llave error

plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de perdida durante el entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('training loss')
plt.legend('training loss')

#prediccion

Temp_C = 0
Temp_F = model.predict([Temp_C])
print (Temp_F)


#Para optimizar cambiamos el Adam a por ejemplo 1 y probamos cual puede ser más certero
