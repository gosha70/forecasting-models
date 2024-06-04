# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import os
import numpy as np
import warnings
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress specific TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


from .base_model_factory import BaseModelFactory, END_EVENT_SEQUENCE

OPTIMIZER = 'adam'
UNITS_PARAM = 'units'
DENSE_UNITS_PARAM = 'dense_units'
INPUT_LENGTH_PARAM = 'input_length'
DEFAULT_EPOCH = 5
DEFAULT_BATCH_SIZE = 32

# Hyperparameters for Classification LSTM
CLASSIFICATION_UNITS = 128
CLASSIFICATION_DROPOUT = 0.5
CLASSIFICATION_OUTPUT_DIM = 64
CLASSIFICATION_LOSS = 'sparse_categorical_crossentropy'
CLASSIFICATION_ACTIVATION = 'softmax'
METRICS = ['accuracy']
CLASSIFICATION_DEFAULT_PARAMS = {
    UNITS_PARAM: CLASSIFICATION_UNITS,
    DENSE_UNITS_PARAM: 0,
    INPUT_LENGTH_PARAM: 0
}

# Hyperparameters for Regression LSTM
REGRESSION_UNITS = 64
REGRESSION_LOSS = 'mse'
REGRESSION_ACTIVATION = 'relu'
INPUT_SHAPE_PARAM = 'input_shape'
REGRESSION_DEFAULT_PARAMS = {
    UNITS_PARAM: REGRESSION_UNITS,
    INPUT_SHAPE_PARAM: 0
}

class LSTM_ModelFactory(BaseModelFactory):
    def __init__(self):
        super().__init__()
        self._ml_model = None
        self.unique_events_count = 0

    def create_classification_model(self, **kwargs):
        params = {**CLASSIFICATION_DEFAULT_PARAMS, **kwargs}

        units = params[UNITS_PARAM]

        # Check required arguments        
        dense_units = params[DENSE_UNITS_PARAM]
        if dense_units == 0:
            raise ValueError(DENSE_UNITS_PARAM)
        input_len = params[INPUT_LENGTH_PARAM]
        if input_len == 0:
            raise ValueError(INPUT_LENGTH_PARAM)

        # Build the LSTM model
        model = Sequential()
        model.add(Embedding(dense_units, CLASSIFICATION_OUTPUT_DIM, input_length=input_len))
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(CLASSIFICATION_DROPOUT))
        model.add(LSTM(units))
        model.add(Dense(dense_units, activation=CLASSIFICATION_ACTIVATION))

        # Compile the model
        model.compile(optimizer=OPTIMIZER, loss=CLASSIFICATION_LOSS, metrics=METRICS)

        return model
        
    def create_regression_model(self, **kwargs):
        params = {**REGRESSION_DEFAULT_PARAMS, **kwargs}
        input_shape = params[INPUT_SHAPE_PARAM]
        if input_shape == 0:
            raise ValueError(INPUT_SHAPE_PARAM)

        model = Sequential([
            LSTM(REGRESSION_UNITS, activation=REGRESSION_ACTIVATION, input_shape=input_shape, return_sequences=True),
            LSTM(REGRESSION_UNITS, activation=REGRESSION_ACTIVATION),
            Dense(1)
        ])
        model.compile(optimizer=OPTIMIZER, loss=REGRESSION_LOSS)
    
        return model
    
    def train(self, X, y):
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Train the model
        self._ml_model.fit(X_train, y_train, epochs=DEFAULT_EPOCH, batch_size=DEFAULT_BATCH_SIZE, validation_split=0.2)

        # Evaluate the model
        return self._ml_model.evaluate(X_test, y_test)
    
    def train_event_sequence(
            self, 
            event_sequences,
            unique_events_count,
            additional_features, 
            is_random_seq: bool):          
        self.unique_events_count = unique_events_count 
        if additional_features is None or len(additional_features) == 0:
            X, y = self.train_simple_event_sequence(event_sequences, is_random_seq)
        else:
            X, y = self.train_complex_event_sequence(event_sequences, additional_features, is_random_seq)

        # Create the classification model
        self._ml_model = self.create_classification_model(
            dense_units=self.unique_events_count + 1,  # +1 for the special end/completed case
            input_length=self._sequence_length
        )

        loss, accuracy = self.train(X, y)

        return loss, accuracy
        
    def train_complex_event_sequence( 
            self, 
            event_sequences,
            additional_features,
            is_random_seq: bool): 
        # Not implented yet
        return self.train_simple_event_sequence(event_sequences, is_random_seq)
        
    def train_simple_event_sequence( 
            self, 
            event_sequences,
            is_random_seq: bool): 
        # Prepare X and y datasets
        X = []
        y = []

        for _, seq in enumerate(event_sequences):
            len_seq = len(seq)
            if len_seq < 2:
                continue
                
            # Generate a random length for the subsequence
            if is_random_seq:
                random_length = random.randint(1, len_seq - 1)
            else:
                random_length = len_seq - 1  

            combined_seq = seq[:random_length]

            # Define the target value
            x_y = seq[random_length] if random_length < len(seq) - 1 else self.unique_events_count

            X.append(combined_seq)
            y.append(x_y)
            #print(f"seq: {seq} X: {combined_seq} -> y: {x_y}")
        
        # Pad sequences
        self._sequence_length = max(len(seq) for seq in X)
        X = pad_sequences(X, maxlen=self._sequence_length, dtype='float32')
        y = np.array(y, dtype='int32')

        return X, y
    
    def predict(self, X):
        in_progress_sequence = pad_sequences([X], maxlen=self._sequence_length, dtype='float32')
        return np.argmax(self._ml_model.predict(in_progress_sequence), axis=-1)[0]   

    
    def predict_duration(self, X):
        in_progress_sequence = pad_sequences([X], maxlen=self._sequence_length, dtype='float32')
        return self._ml_model.predict(in_progress_sequence)[0][0]      
    
    def train_duration_sequencOLD(
        self, 
        event_sequences,
        duration_sequences,
        unique_events_count,
        additional_features, 
        is_random_seq: bool): 
        # Prepare X and y datasets
        X = []
        y = []

        for seq, durations in zip(event_sequences, duration_sequences):
            len_seq = len(seq)
            if len_seq < 2:
                continue

            # Generate a random length for the subsequence
            if is_random_seq:
                random_length = random.randint(1, len_seq - 1)
            else:
                random_length = len_seq - 1

            combined_seq = seq[:random_length]

            # Define the target value (duration)
            target_duration = durations[random_length] if random_length < len(durations) else 0

            X.append(combined_seq)
            y.append(target_duration)
        
        # Pad sequences
        self._sequence_length = max(len(seq) for seq in X)
        X = pad_sequences(X, maxlen=self._sequence_length, dtype='float32')
        y = np.array(y, dtype='float32')

        # Reshape X to have a third dimension
        X = np.expand_dims(X, axis=-1)

        # Create the regression model
        self._ml_model = self.create_regression_model(
            input_shape=(self._sequence_length, 1)
        )

        loss, mse = self.train(X, y)

        return loss, mse
    
    def train_duration_sequence(
        self, 
        event_sequences,
        duration_sequences,
        unique_events_count,
        additional_features, 
        is_random_seq: bool): 
        # Prepare X and y datasets
        X = []
        y = []

        for seq, durations in zip(event_sequences, duration_sequences):
            len_seq = len(seq)
            if len_seq < 2:
                continue

            # Generate a random length for the subsequence
            if is_random_seq:
                random_length = random.randint(1, len_seq - 1)
            else:
                random_length = len_seq - 1

            combined_seq = seq[:random_length]

            # Define the target value (duration)
            target_duration = durations[random_length] if random_length < len(durations) else 0

            X.append(combined_seq)
            y.append(target_duration)
        
        # Normalize the target durations
        y = np.array(y, dtype='float32')
        y_mean = np.mean(y)
        y_std = np.std(y)
        y = (y - y_mean) / y_std

        # Pad sequences
        self._sequence_length = max(len(seq) for seq in X)
        X = pad_sequences(X, maxlen=self._sequence_length, dtype='float32')
        
        # Reshape X to have a third dimension
        X = np.expand_dims(X, axis=-1)

        # Create the regression model
        self._ml_model = self.create_regression_model(
            input_shape=(self._sequence_length, 1)
        )

        loss, mse = self.train_prediction(X, y)

        return loss, mse

    def train_prediction(self, X, y):
        history = self._ml_model.fit(
            X, y,
            epochs=5,
            validation_split=0.2
        )
        final_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        return final_loss, val_loss
