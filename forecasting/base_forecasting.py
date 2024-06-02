# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from abc import ABC, abstractmethod

from models.base_model_factory import BaseModelFactory
from forecasting.config.forecasting_config import ForecastingConfig
from prep.config.prep_config import PrepConfig
from .forecasting_type import ForecastingType

class BaseForecasting(ABC):
    """The abstract class for predictive analytics cases"""
  
    def __init__(self, forecasting_config: ForecastingConfig, prep_config: PrepConfig):
        """Initializes BaseFileConverter with optional Language and Logging."""
        self._forecasting_config = forecasting_config
        self._prep_config = prep_config

    @property
    def forecasting_config(self)-> ForecastingConfig:
        return self._forecasting_config
    
    @property
    def prep_config(self)-> PrepConfig:
        return self._prep_config
    
    @abstractmethod
    def get_forecasting_columns(self):
        return []

    @abstractmethod
    def train(self, model: BaseModelFactory):
        raise ValueError('BaseForecasting', 'train()')
    
    @abstractmethod
    def predict(self, ml_model, X):
        raise ValueError('BaseForecasting', 'predict()')
      
    @staticmethod
    def create_prediction_training(forecasting_config: ForecastingConfig, prep_config: PrepConfig, dataset):
        tp = forecasting_config.forecasting_type
        if tp == ForecastingType.NEXT_EVENT:
            from .next_event_forecasting import NextEventForecasting
            return NextEventForecasting(forecasting_config, prep_config, dataset)
        elif tp == ForecastingType.CASE_DURATION:
            from .case_duration_forecasting import CaseDurationForecasting
            return CaseDurationForecasting(forecasting_config, prep_config, dataset)
        elif tp == ForecastingType.EVENT_DURATION:
            from .event_duration_forecasting import EventDurationForecasting
            return EventDurationForecasting(forecasting_config, prep_config, dataset)
        elif tp == ForecastingType.CASE_CLASS:
            from .case_class_prediction import CaseClassForecasting
            return CaseClassForecasting(forecasting_config, prep_config, dataset)
        else:
            raise ValueError(f"Unknown PredictionType: {tp}")
        