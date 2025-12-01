from io import StringIO
import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, LayerNormalization
# from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# from keras.utils import np_utils
import numpy as np

#  we need (n , 30, 34)



import pandas as pd
NotSingleLeg = pd.read_csv(r'C:\Projects\AI\Image_class_bjj\Pose\Dataset\Pose_notSingleLeg_points.csv', header=0)
data=NotSingleLeg.to_records(index=False)
print(data)
