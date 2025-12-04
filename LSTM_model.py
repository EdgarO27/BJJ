from io import StringIO
import os
# from tensorflow.python.keras.models import Sequential
# # from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# from keras.utils import np_utils
# from keras.layers import LSTM
import numpy as np
import csv
import tensorflow as tf
from keras import layers
from keras import utils
from keras import models
from keras import Sequential
from keras import KerasTensor
import datetime
import matplotlib.pyplot as plt
import pandas as pd


single_leg = np.load(r'C:\Projects\AI\Image_class_bjj\Single_leg.npy')
not_single_leg = np.load(r'C:\Projects\AI\Image_class_bjj\Not_Single_leg.npy')

final_dataset = np.concatenate([single_leg,not_single_leg]) #Combined data

# print(single_leg.shape)
# print(not_single_leg.shape)
# print(final_dataset.shape)

Single_label = np.ones((102460), dtype="int").reshape(-1,1)         #Make labels y 
Not_Single_label = np.zeros((22703), dtype="int").reshape(-1,1)



# labels = np.vstack([Single_label],[Not_Single_label]) #Make label stack based on how final data is set up 

# ohe = OneHotEncoder()

# y = ohe.fit_transform(labels).toarray().astype(int)

# print(y[0:5])

print(Single_label)
print(Not_Single_label)

# # Train Test Split
# X_train, X_test, y_train, y_test = train_test_split(final_dataset,y,test_size =0.20, shuffle =True)


# log_dir = os.path.join("Logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tb_callback = TensorBoard(log_dir=log_dir)
# print(X_train.shape)
# print(X_test.shape)

# actions = np.array(['Single_Leg', 'Not_Single_Leg'])

# model = Sequential()
# model.add(layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30,102), batch_size= 30))
# model.add(layers.Dense())
# model.add(layers.Dense(actions.shape[0], activation='softmax'))

# model.summary()

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# # mod = models.load_model('SingleLeg_classifier3.h5')
# model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10)


# model.save('SingleLeg_classifier3.keras')  # creates a HDF5 file 'my_model.h5'

