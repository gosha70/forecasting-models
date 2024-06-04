
# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import Any, Dict
import pandas as pd

from .transformation_config import TransformationConfig
from .transformation_type import TransformationType
from .event_transformation  import EventTransformation
from .case_event_transformation import CaseEventTransformation

class TransformationManager:
    def __init__(
            self, 
            transformation_config: TransformationConfig, 
            dataset_one: pd.DataFrame,
            dataset_two: pd.DataFrame):        
        self._transformation_config = transformation_config   
        self._dataset_one = dataset_one 
        self._dataset_two = dataset_two
     
    @property
    def transformation_config(self)-> TransformationConfig:
        return self._transformation_config   
    
    @property
    def first_dataset(self):
        return self._dataset_one
    
    @property
    def second_dataset(self):
        return self._dataset_two
    
    def execute(self) -> pd.DataFrame:
        transformation = None
        if self.transformation_config.transformation_type == TransformationType.EVENTS:
            transformation = EventTransformation(
                transformation_config=self.transformation_config, 
                dataset=self.first_dataset
            )
        elif self.transformation_config.transformation_type == TransformationType.CASES_EVENTS:
            transformation = CaseEventTransformation(
                transformation_config=self.transformation_config, 
                case_dataset=self.first_dataset,
                event_dataset=self.second_dataset
            )
        else:
            raise Exception("Transformation type not supported")
        
        return transformation.execute()
