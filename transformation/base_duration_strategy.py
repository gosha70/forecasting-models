# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

from .duration_strategy_type import DurationStrategyType


DURATION_STRATEGY_CONFIG_PROP = 'duration_strategy'
DURATION_STRATEGY_TYPE_PROP = 'type'

"""
Base class for all duration strategies.
"""
class BaseDurationStrategy(ABC):
    def __init__(self, type: DurationStrategyType):
        super().__init__()
        self._type = type

    def get_sort_by_column(self)-> str:
        raise ValueError('BaseDurationStrategy', 'get_sort_by_column()')    
    
    def calculate_duration(
            self, 
            dataset: pd.DataFrame, 
            max_length: int,
            name_prefix: str) -> pd.DataFrame:  
        raise ValueError('BaseDurationStrategy', 'calculate_duration()')  
        
    @property
    def type(self) -> DurationStrategyType:
        return self._type
    
    @staticmethod
    def create_duration_strategy(transformation_config: Dict[str, Any]):
        duration_config = transformation_config[DURATION_STRATEGY_CONFIG_PROP]
        tp = int(duration_config[DURATION_STRATEGY_TYPE_PROP])
        if tp == DurationStrategyType.SINGLE_TIMESTAMP.value:
            from .single_timestamp_duration_strategy import SingleTimestampDurationStrategy
            return SingleTimestampDurationStrategy(duration_config)
        elif tp == DurationStrategyType.START_END_TIMESTAMP.value:
            from .start_end_duration_strategy import StartEndDurationStrategy
            return StartEndDurationStrategy(duration_config)
        else:
            raise ValueError(f"Unknown Duration Strategy: {tp}")
        