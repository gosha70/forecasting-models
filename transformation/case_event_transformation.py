# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import pandas as pd

from.transformation_config import TransformationConfig

class CaseEventTransformation:
    def __init__(
            self, 
            transformation_config: TransformationConfig, 
            case_dataset: pd.DataFrame,
            event_dataset: pd.DataFrame):
        self._config = transformation_config
        self._case_dataset = case_dataset
        self._event_dataset = event_dataset

    @property
    def config(self) -> TransformationConfig:
        return self._config    
    
    @property
    def case_dataset(self) -> pd.DataFrame:
        return self._case_dataset    
    
    @property
    def event_dataset(self) -> pd.DataFrame:
        return self._event_dataset    

    def execute(self) -> pd.DataFrame:  
        raise ValueError('CaseEventTransformation', 'execute()')    
