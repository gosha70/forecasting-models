# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import numpy as np
import pandas as pd

from models.base_model_factory import BaseModelFactory, END_EVENT_SEQUENCE
from prep.config.prep_config import PrepConfig
from prep.data_prep import DataPrep
from forecasting.base_forecasting import BaseForecasting
from forecasting.config.forecasting_config import ForecastingConfig

class NextEventForecasting(BaseForecasting):
    def __init__(
            self, 
            forecasting_config: ForecastingConfig, 
            prep_config: PrepConfig, 
            dataset: pd.DataFrame):
        super().__init__(forecasting_config, prep_config)
        self.event_columns = dataset.columns[dataset.columns.str.contains(forecasting_config.event_column_name_pattern.replace('*', '.*'))] 
        print(f"Time series columns: {self.event_columns}")
        self.unique_events = None
        self.event_to_idx = None
        self.idx_to_event = None
        self.additional_features = None
        self._prep_config = prep_config
        self._dataset = dataset

    def get_forecasting_columns(self):
        return self.event_columns 

    def prepare(self, prep_config: PrepConfig, dataset):        
        prep_task = DataPrep(prep_config, dataset.copy())
        return prep_task.prepare(ignore_columns=self.event_columns) 

    def preprocess_data(self, dataset: pd.DataFrame):
        # Identify additional features (columns not starting with the event prefix)
        self.additional_features = [col for col in dataset.columns if col not in self.event_columns]

        # Extract the sequences and handle missing values
        sequences = dataset[self.event_columns].fillna('').values
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
        return [seq for seq in numerical_sequences if all(idx != -1 for idx in seq)]
    
    def train(self, model_factory: BaseModelFactory):
        prep_task = DataPrep(prep_config=self._prep_config, dataset=self._dataset.copy())
        df = prep_task.prepare(preserved_columns=self.event_columns) 
       
        valid_sequences = self.preprocess_data(df)

        print(f"Number of sequences: {len(valid_sequences)}")

        return  model_factory.train_event_sequence(
            event_sequences=valid_sequences, 
            unique_events_count=len(self.unique_events),
            additional_features=self.additional_features, 
            is_random_seq=self.forecasting_config.is_random_seq)

    def predict(self, model_factory: BaseModelFactory, X):
        in_progress_sequence = [self.event_to_idx.get(event, len(self.unique_events)) for event in X]
        predicted_event_idx = model_factory.predict(in_progress_sequence)
        return self.idx_to_event.get(predicted_event_idx, END_EVENT_SEQUENCE)

    def predict_proba(self, model_factory: BaseModelFactory, X, top_k=None, threshold=None):
        in_progress_sequence = [self.event_to_idx.get(event, len(self.unique_events)) for event in X]
        raw_proba = model_factory.predict_proba(in_progress_sequence)

        proba = {}
        for idx, prob in raw_proba.items():
            event_name = self.idx_to_event.get(idx, END_EVENT_SEQUENCE)
            proba[event_name] = prob

        if threshold is not None:
            proba = {e: p for e, p in proba.items() if p >= threshold}

        sorted_proba = dict(sorted(proba.items(), key=lambda x: x[1], reverse=True))

        if top_k is not None:
            sorted_proba = dict(list(sorted_proba.items())[:top_k])

        return sorted_proba
        