import os
import cv2
# import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf

from collections import deque
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

import keras

from keras import layers
from keras import utils
from keras import models


seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Specify the height and width to which each video frame will be resized in our dataset.


# Specify the number of frames of a video that will be fed to the model as one sequence.
# Since the LSTM works upon the sequential dataset
SEQUENCE_LENGTH = 10

# Specify the directory containing the UCF50 dataset.
DATASET_DIR = "Single_Leg"

# Specify the list containing the names of the classes used for training.
CLASSES_LIST = ["Single Leg"]
def frames_extraction(folder):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []
    features = []
    labels = []
    IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
    
    # Iterate through the Video Frames.
    for image in os.listdir(folder):

        # Set the current frame position of the video.
        # video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        
        # Reading the frame from the video.
        image = os.path.join(folder, image)
        frame = cv2.imread(image)

        # Check if Video frame is not successfully read then break the loop
        


        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Append the normalized frame into the frames list
        labels.append(1)
        frames_list.append(normalized_frame)

    # Release the VideoCapture object.
    cv2.destroyAllWindows()
    features = np.asarray(frames_list)
    labels = np.array(labels)
    # Return the frames list.
    return features, labels






#Splitting data and one hot encode 
def conversion(features, label):
    one_hot_encoded_labels = utils.to_categorical(label, num_classes=2)
    # Split the Data into Train ( 75% ) and Test Set ( 25% ).
    
    features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                            test_size = 0.25, shuffle = True,
                                                                            random_state = seed_constant)
    return features_train, labels_train, features_test, labels_test

def batch_frames_to_sequences(frames, labels, sequence_length=10):
    total = len(frames) // sequence_length
    frames = frames[:total * sequence_length]
    labels = labels[:total * sequence_length]
    sequences = frames.reshape((total, sequence_length, 64, 64, 3))
    sequence_labels = labels[::sequence_length]
    return sequences, sequence_labels

    
def create_LRCN_model(IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE):
    '''
    This function will construct the required LRCN model.
    Returns:
        model: It is the required constructed LRCN model.
    '''

    # We will use a Sequential model for model construction.
    model = models.Sequential()

    # Define the Model Architecture.
    ########################################################################################################################

    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQUENCE, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))

    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))

    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))

    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))

    model.add(layers.TimeDistributed(layers.Flatten()))

    model.add(layers.LSTM(32))

    model.add(layers.Dense(2, activation = 'sigmoid'))

    ########################################################################################################################

    # Display the models summary.
    model.summary()

    # Return the constructed LRCN model.
    return model



def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    '''
    This function will plot the metrics passed to it in a graph.
    Args:
        model_training_history: A history object containing a record of training and validation
                                loss values and metrics values at successive epochs
        metric_name_1:          The name of the first metric that needs to be plotted in the graph.
        metric_name_2:          The name of the second metric that needs to be plotted in the graph.
        plot_name:              The title of the graph.
    '''

    # Get metric values using metric names as identifiers.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_value_1))

    # Plot the Graph.
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()



