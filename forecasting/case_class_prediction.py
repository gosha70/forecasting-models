# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from forecasting.base_forecasting import BaseForecasting
from models.base_model_factory import BaseModelFactory
from forecasting.config.forecasting_config import ForecastingConfig
from prep.config.prep_config import PrepConfig

class CaseClassForecasting(BaseForecasting):
    """Class for predicting the class of a case"""

    def __init__(self, forecasting_config: ForecastingConfig, prep_config: PrepConfig, dataset):
        super().__init__(forecasting_config, prep_config)
        self.dataset = dataset

    def get_forecasting_columns(self):
        return self.forecasting_config.target_columns

    def train(self, model: BaseModelFactory):
        X_train = self.dataset[self.forecasting_config.feature_columns]
        y_train = self.dataset[self.forecasting_config.target_columns]
        model.train(X_train, y_train)

    def predict(self, ml_model, X):
        return ml_model.predict(X)