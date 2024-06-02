# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.base_model_factory import BaseModelFactory
from prep.config.prep_config import PrepConfig
from prep.data_prep import DataPrep
from forecasting.base_forecasting import BaseForecasting
from forecasting.config.forecasting_config import ForecastingConfig

class NextEventForecasting(BaseForecasting):
    def __init__(self, forecasting_config: ForecastingConfig, prep_config: PrepConfig, dataset):
        super().__init__(forecasting_config, prep_config)
        self.sequence_length = 0
        self.event_columns = [col for col in dataset.columns if col in forecasting_config.event_columns]
        print(f"Time series columns: {self.event_columns}")
        self.unique_events = None
        self.event_to_idx = None
        self.idx_to_event = None
        self.additional_features = None
        self.df, self.scalers = self.prepare(prep_config, dataset)

    def prepare(self, prep_config: PrepConfig, dataset):        
        prep_task = DataPrep(prep_config, dataset.copy())
        return prep_task.prepare(ignore_columns=self.event_columns)  

    def get_special_columns(self):
        return self.event_columns   

    def preprocess_data(self):
        # Identify additional features (columns not starting with the event prefix)
        self.additional_features = [col for col in self.df.columns if col not in self.event_columns]

        # Extract the sequences and handle missing values
        sequences = self.df[self.event_columns].fillna('').values
        sequences = [list(filter(None, seq)) for seq in sequences if any(seq)]
        
        # Create mappings for events to indices
        all_events = [event for seq in sequences for event in seq]
        self.unique_events = np.unique(all_events)
        self.event_to_idx = {event: idx for idx, event in enumerate(self.unique_events)}
        self.idx_to_event = {idx: event for event, idx in self.event_to_idx.items()}

        # Print debug info
        print(f"Unique events: {self.unique_events}")
        print(f"Event to index mapping: {self.event_to_idx}")

        # Handle cases where the event might not be in event_to_idx
        def safe_map(event):
            return self.event_to_idx.get(event, -1)

        # Convert events to indices
        numerical_sequences = [[safe_map(event) for event in seq] for seq in sequences]

        # Ensure no invalid indices
        for seq in numerical_sequences:
            if any(idx == -1 for idx in seq):
                print(f"Invalid sequence found: {seq}")

        # Filter out sequences with invalid indices
        valid_sequences = [seq for seq in numerical_sequences if all(idx != -1 for idx in seq)]

        # Prepare X and y datasets
        X = []
        y = []
        additional_features_data = self.df[self.additional_features].values
        
        for i, seq in enumerate(valid_sequences):
            for j in range(len(seq)):
                combined_seq = seq[:j+1] + list(additional_features_data[i])
                X.append(combined_seq)
                y.append(seq[j] if j < len(seq) - 1 else len(self.unique_events))
        
        # Pad sequences
        self.sequence_length = max(len(seq) for seq in X)
        X = pad_sequences(X, maxlen=self.sequence_length, dtype='float32')
        y = np.array(y, dtype='int32')

        # Debugging output
        print(f"Preprocessed X shape: {X.shape}")
        print(f"Preprocessed y shape: {y.shape}")
        
        return X, y

    def train(self, model_factory: BaseModelFactory):
        X, y = self.preprocess_data()
        
        # Create the classification model
        ml_model = model_factory.create_classification_model(
            dense_units=len(self.unique_events) + 1,  # +1 for the special end/completed case
            input_length=self.sequence_length
        )

        loss, accuracy = model_factory.train(ml_model, X, y)

        return ml_model, loss, accuracy

    def predict(self, ml_model, X):
        # Assume X is a dictionary with event sequence and additional features
        event_sequence = X['event_sequence']
        additional_features = X['additional_features']

        # Encode and normalize the additional features using the stored scalers
        for i, feature in enumerate(additional_features):
            col_name = self.additional_features[i]
            scaler = self.scalers[col_name]
            additional_features[i] = scaler.transform(np.array(feature).reshape(-1, 1)).flatten()[0]

        in_progress_sequence = [self.event_to_idx.get(event, len(self.unique_events)) for event in event_sequence]
        combined_sequence = in_progress_sequence + additional_features
        combined_sequence = pad_sequences([combined_sequence], maxlen=self.sequence_length, dtype='float32')
        predicted_event_idx = np.argmax(ml_model.predict(combined_sequence), axis=-1)[0]

        return self.idx_to_event.get(predicted_event_idx, 'END')