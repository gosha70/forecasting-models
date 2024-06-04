# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import Dict, Any

from forecasting.forecasting_type import ForecastingType

FORECASTING_TYPE_PROP = 'forecasting_type'
EVENT_COLUMN_PATTERN_PROP ='event_column_pattern' #The name pattern for event columns
DURATION_COLUMN_PATTERN_PROP ='duration_column_pattern' 
RANDOM_SEQ_PROP = 'random_sequence'

class ForecastingConfig:
    def __init__(
            self, 
            forecasting_type: ForecastingType, 
            name_pattern: str, 
            duration_pattern: str,
            is_random_seq: bool):        
        self._type = forecasting_type   
        self._event_column_name_pattern = name_pattern
        self._duration_column_name_pattern = duration_pattern
        self._is_random_seq = is_random_seq
        print(f"forecasting type: {self._type}; event column name pattern: {self._event_column_name_pattern}; is random sequence: {self._is_random_seq}")
     
    @property
    def forecasting_type(self)-> ForecastingType:
        return self._type   
    
    @property
    def event_column_name_pattern(self)-> str:
        return self._event_column_name_pattern    

    @property
    def duration_column_name_pattern(self)-> str:
        return self._duration_column_name_pattern 
    
    @property
    def is_random_seq(self)-> bool:
        return self._is_random_seq
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        type = ForecastingType[data[FORECASTING_TYPE_PROP].upper()]
        event_column_pattern = data[EVENT_COLUMN_PATTERN_PROP]
        duration_column_pattern = data.get(DURATION_COLUMN_PATTERN_PROP, None)
        is_random_seq = data[RANDOM_SEQ_PROP]
        return cls(
            forecasting_type=type, 
            name_pattern=event_column_pattern, 
            duration_pattern=duration_column_pattern, 
            is_random_seq=is_random_seq
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            FORECASTING_TYPE_PROP: self.forecasting_type.value,
            EVENT_COLUMN_PATTERN_PROP: self.event_column_name_pattern,
            DURATION_COLUMN_PATTERN_PROP: self.duration_column_name_pattern,
            RANDOM_SEQ_PROP: self.is_random_seq
        }