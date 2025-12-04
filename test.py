import numpy as np
import csv
import tensorflow as tf
from keras import layers
from keras import utils
from keras import models
from keras import Sequential
from keras import KerasTensor
import datetime

import pandas as pd



model = models.load_model('SingleLeg_classifier1.h5')

print(model.summary())