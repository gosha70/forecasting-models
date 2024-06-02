# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


from .base_model_factory import BaseModelFactory

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
REGRESSION_UNITS = 50
REGRESSION_LOSS = 'mean_squared_error'
REGRESSION_ACTIVATION = 'relu'
INPUT_SHAPE_PARAM = 'input_shape'
REGRESSION_DEFAULT_PARAMS = {
    UNITS_PARAM: REGRESSION_UNITS,
    INPUT_SHAPE_PARAM: 0
}

class LSTM_ModelFactory(BaseModelFactory):
    def __init__(self):
        super().__init__()

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
    
    def train(self, ml_model, X, y):
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Train the model
        ml_model.fit(X_train, y_train, epochs=DEFAULT_EPOCH, batch_size=DEFAULT_BATCH_SIZE, validation_split=0.2)

        # Evaluate the model
        return ml_model.evaluate(X_test, y_test)