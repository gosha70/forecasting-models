# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import Dict, Any

from models.config.model_config import ModelConfig
from prep.config.prep_config import PrepConfig
from .test_data import TestData

PREP_CONFIG_PROP = 'prep_config'
MODEL_CONFIG_PROP = 'model_config'
TEST_DATA_PROP = 'test_data'

class TrainConfig:
    def __init__(
            self, 
            prep_config: PrepConfig, 
            model_config: ModelConfig,
            test_data: TestData):
        self._prep_config = prep_config
        self._model_config = model_config
        self._test_data = test_data
                               
    @property
    def prep_config(self)-> PrepConfig:
        return self._prep_config        
                 
    @property
    def model_config(self)-> ModelConfig:
        return self._model_config  
     
    @property
    def test_data(self)-> TestData:
        return self._test_data     

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        prep_config = PrepConfig.from_dict(data[PREP_CONFIG_PROP])
        model_config = ModelConfig.from_dict(data[MODEL_CONFIG_PROP])
        test_data = TestData(test_data=data[TEST_DATA_PROP])
        return cls(prep_config, model_config, test_data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            PREP_CONFIG_PROP: self.prep_config.to_dict(),
            MODEL_CONFIG_PROP: self.model_config.to_dict(),
            TEST_DATA_PROP: self._test_data.test_data

        }