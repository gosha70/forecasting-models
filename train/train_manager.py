# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import numpy as np
import pandas as pd

from utils.file_util import load_json
from train.config.train_config import TrainConfig
from forecasting.base_forecasting import BaseForecasting

class TrainManager:
    def __init__(self, train_config: TrainConfig, dataset: pd.DataFrame):        
        self._train_config = train_config   
        self._dataset = dataset
     
    @property
    def train_config(self)-> TrainConfig:
        return self._train_config   
    
    @property
    def dataset(self):
        return self._dataset

    def dataset_report(self):
        # Summary statistics and data types
        print(self.dataset.describe())
        print(self.dataset.info())
        # Check data types of all columns
        data_types = self.dataset.dtypes

        # Confirm all are numeric
        all_numeric = data_types.apply(lambda x: np.issubdtype(x, np.number)).all()

        print("Data types of all columns after encoding:")
        print(data_types)
        print("\nAre all columns numeric? ", "Yes" if all_numeric else "No")

    def execute(self):
        self.dataset_report()

        forecasting_config = self.train_config.model_config.forecasting_config
        prep_config = self.train_config.prep_config

        prediction_task = BaseForecasting.create_prediction_training(
            forecasting_config=forecasting_config,
            prep_config=prep_config, 
            dataset=self.dataset
        )
        model_factory = self.train_config.model_config.create_model()
        loss, accuracy = prediction_task.train(model_factory)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

        # Example: in-progress events is the list of observed events for the in-progress case
        test_data = self.train_config.test_data.get_X_y_pairs()
        for in_progress_events, expected_result in test_data:
            predicted_event = prediction_task.predict(model_factory=model_factory, X=in_progress_events)
            print(f'The predicted next event is: {predicted_event}; expected result: {expected_result}')