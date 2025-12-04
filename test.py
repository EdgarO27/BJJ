import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras import Input # Import Input explicitly
import os

def load_data(x_train_path, y_train_path, x_val_path, y_val_path):
    """
    Loads training and validation data from NumPy files.

    Args:
        x_train_path (str): Path to the training pose sequences (.npy).
        y_train_path (str): Path to the training labels (.npy).
        x_val_path (str): Path to the validation pose sequences (.npy).
        y_val_path (str): Path to the validation labels (.npy).
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val) as NumPy arrays.
    """
    print("Loading data from disk...")
    try:
        X_train = np.load(x_train_path).astype(np.float32)
        y_train = np.load(y_train_path).astype(np.float32)
        X_val = np.load(x_val_path).astype(np.float32)
        y_val = np.load(y_val_path).astype(np.float32)
        
        # Ensure labels are 2D for Keras (N, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, axis=-1)
        if y_val.ndim == 1:
            y_val = np.expand_dims(y_val, axis=-1)

        print(f"Data loaded successfully:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_val shape:   {X_val.shape}")
        
        return X_train, y_train, X_val, y_val
        
    except FileNotFoundError as e:
        print(f"Error: One or more data files not found. Check path: {e.filename}")
        # Return empty arrays to prevent immediate crash
        return np.array([]), np.array([]), np.array([]), np.array([])
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])


def build_lstm_model(input_shape):
    """
    Builds a Sequential Keras model using LSTM layers for pose sequence classification.
    """
    
    # Define the model architecture
    model = Sequential([
        Input(shape=input_shape), 
        LSTM(128, return_sequences=True, name='lstm_1'),
        Dropout(0.5, name='dropout_1'),
        LSTM(64, name='lstm_2'),
        Dropout(0.5, name='dropout_2'),
        Dense(1, activation='sigmoid', name='output_classifier')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --- Main Script Usage ---

# 1. Define input parameters based on your data: (30, 102)
TIMESTEPS = 30
FEATURES = 102
INPUT_SHAPE = (TIMESTEPS, FEATURES)

# --- Define the paths to your NPY files here ---
# NOTE: Replace these placeholder paths with the actual paths to your data
X_TRAIN_PATH = 'data/X_train_sequences.npy'
Y_TRAIN_PATH = 'data/y_train_labels.npy'
X_VAL_PATH = 'data/X_val_sequences.npy'
Y_VAL_PATH = 'data/y_val_labels.npy'
# ------------------------------------------------

# 2. Load the actual data
X_train, y_train, X_val, y_val = load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH)

# Check if data loading failed or resulted in empty arrays
if X_train.size == 0 or X_val.size == 0:
    print("\nTraining aborted due to missing or empty data files.")
else:
    # 3. Build the model
    model = build_lstm_model(INPUT_SHAPE)

    # Display the model summary to confirm the layers and parameters
    model.summary()

    print("\n--- Model Training ---")

    # Define Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    # 4. Train the model using the loaded data
    history = model.fit(
        X_train, 
        y_train,
        epochs=100, 
        batch_size=32, 
        validation_data=(X_val, y_val), # Use your actual validation data
        callbacks=[early_stop], 
        verbose=1
    )

    # 5. Once trained, you can save the model weights:
    # model.save('pose_classifier_weights.h5')
    # print("\nModel saved successfully!")