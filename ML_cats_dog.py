# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:31:28 2020

@author: Sys10
"""


import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#En caso de hacer unzip al archivo desde el enlace https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

#dataset_path = "./cats_and_dogs_filtered.zip"
#zip_object = zipfile.ZipFile(file=dataset_path, mode ="r")
#zip_object.extractall("./")
#zip_object.close()

#hacemos el path
dataset_path = "./cats_and_dogs_filtered"
#path del set de entrenamiento
train_directory = os.path.join(dataset_path, "train")
validation_dir = os.path.join(dataset_path, "validation")

#armando el modelo, primero definimos el tamaño de la imagen y a colores es decir rgb
img_shape = (128,128,3)
#Para el modelo no queremos que mobileNet use su propio output por lo tanto el include top sera False
#los pesos que queremos traer son imagenet (la red entrenada por google)
model = tf.keras.applications.MobileNetV2(input_shape = img_shape, include_top = False, weights = "imagenet")

#congelamos el modelo base que trae mobilenet
base_model.trainable = False

#Definimos la cabeza
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

#definimos el output
prediction_layer = tf.keras.layers.Dense(units = 1, activation = 'sigmoid')(global_average_layer)

#definimos el modelo
model = tf.keras.models.Model(inputs=base_model.input, outputs =prediction_layer)
#Compilado y optimizacion, en este caso con RMSprop en vez de Adam, dado que es mas aconsejable para este tipo de modelos

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics =['accuracy'])

#Usamos ImageDataGenerator para el set de entrenamiento

data_gen_train = ImageDataGenerator(rescale =1/255.)
data_gen_valid = ImageDataGenerator(rescale =1/255.)


train_gen = data_gen_train.flow_from_directory(train_directory, target_size = (128,128), batch_size =128, class_model = 'binary')
valid_gen = data_gen_valid.flow_from_directory(validation_directory, target_size = (128,128), batch_size =128, class_model = 'binary')

#Entrenando modelo
model.fit_generator(train_gen, epochs = 5, validation_data = valid_gen)

#Evaluando
valid_loss, valid_accuracy = model.evaluate(valid_generator)

#          Afinando para lo cual en base_model.trainable en esta ocasion sera True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at:]:
    layer.trainable = False

#Nuevo compilado
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss= 'binary_crossentropy', metrics= ['accuracy'])

#afinamiento y entrenamiento nuevo
model.fit_generator(train_gen, epochs = 5, validation_data = valid_gen)

#Evaluacion nueva
valid_loss, valid_accuracy = model.evaluate(valid_generator)





