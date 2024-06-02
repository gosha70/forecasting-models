# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import Dict, Any, List

from forecasting.forecasting_type import ForecastingType

FORECASTING_TYPE_PROP = 'forecasting_type'
EVENT_COLUMNS_PROP ='event_columns' #The column name of earliest event must be at the beginning of the  !!!

class ForecastingConfig:
    def __init__(self, forecasting_type: ForecastingType, event_columns: List[str]):        
        self._type = forecasting_type   
        self._event_columns = event_columns
        print(f"forecasting type: {self._type}; event sequence: {self._event_columns}")
     
    @property
    def forecasting_type(self)-> ForecastingType:
        return self._type   
    
    @property
    def event_columns(self)-> str:
        return self._event_columns
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        type = ForecastingType[data[FORECASTING_TYPE_PROP].upper()]
        event_columns = data[EVENT_COLUMNS_PROP]
        return cls(type, event_columns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            FORECASTING_TYPE_PROP: self.forecasting_type.value,
            EVENT_COLUMNS_PROP: self.event_columns
        }