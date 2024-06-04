# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import Dict, Any

from models.base_model_factory import BaseModelFactory
from models.lstm_model_factory import LSTM_ModelFactory
from models.markov_chain_factory import MarkivChainModelFactory
from models.model_type import ModelType
from forecasting.config.forecasting_config import ForecastingConfig

MODEL_TYPE_PROP = 'model_type'
MODEL_PARAM_PROP = 'model_params'
FORECASTING_CONFIG_PROP = 'forecasting_config'

class ModelConfig:
    def __init__(
            self, 
            model_type: ModelType, 
            forecasting_config: ForecastingConfig,
            model_params: Dict[str, Any]):
        self._model_type = model_type
        self._forecasting_config = forecasting_config
        self._model_params = model_params
                 
    @property
    def drop_colum_patterns(self)-> ModelType:
        return self._model_type
    
    @property
    def model_params(self)-> Dict[str, Any]:
        return self._model_params    
                 
    @property
    def forecasting_config(self)-> ForecastingConfig:
        return self._forecasting_config  
    
    def create_model(self)-> BaseModelFactory:
        if  self._model_type == ModelType.LSTM:
            return LSTM_ModelFactory()
        elif self._model_type == ModelType.MC:
            return MarkivChainModelFactory()
        return None   
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            MODEL_TYPE_PROP: self._model_type.value,
            MODEL_PARAM_PROP: self.model_params,
            FORECASTING_CONFIG_PROP: self.forecasting_config.to_dict()
        }   

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        model_type = ModelType[data[MODEL_TYPE_PROP].upper()]
        forecasting_config = ForecastingConfig.from_dict(data[FORECASTING_CONFIG_PROP])
        model_params = data[MODEL_PARAM_PROP]
        return cls(model_type, forecasting_config, model_params)