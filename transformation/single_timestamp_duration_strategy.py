# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

from .base_duration_strategy import BaseDurationStrategy
from .duration_strategy_type import DurationStrategyType

SINGLE_TIMESTAMP_PROP = 'time_column'

"""
The class for storing the configuration for calulating the event duration
using a single timestamp column; the duration calculated by substracting
the timestamp of the current event from the timestamp of the next event.
```
   "duration_strategy": {
      "type": 2,
      "time_column": "sys_updated_at"
   }
```   
"""
class SingleTimestampDurationStrategy(BaseDurationStrategy):
    def __init__(self, duration_config: Dict[str, Any]):
        super().__init__(type=DurationStrategyType.SINGLE_TIMESTAMP)
        self._time_column = duration_config[SINGLE_TIMESTAMP_PROP]
   
    def get_sort_by_column(self)-> str:
        return self._time_column       

    @property
    def time_column(self) -> str:
        return self._time_column
    
    def calculate_duration(
            self, 
            dataset: pd.DataFrame, 
            max_length: int,
            name_prefix: str) -> pd.DataFrame:        
        # Calculate duration for each event
        for i in range(max_length - 1):
            dataset[f'{name_prefix}{i+1}'] = dataset[self._time_column].apply(
                lambda x: (x[i+1] - x[i]).total_seconds() if i + 1 < len(x) else None
            )  

        return dataset     