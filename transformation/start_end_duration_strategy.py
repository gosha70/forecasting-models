# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from abc import ABC, abstractmethod
from typing import Any, Dict

from .base_duration_strategy import BaseDurationStrategy
from .duration_strategy_type import DurationStrategyType

START_TIMESTAMP_PROP = 'start_column'
END_TIMESTAMP_PROP = 'end_column'
"""
The class for storing the configuration for calulating the event duration
using twp timestamp columns; the duration calculated by substracting
the timestamp in the `start_column` column from the timestamp
in the `end_column` column.
```
   "duration_strategy": {
      "type": 1,
      "start_column": "Start",
      "end_column": "End"
   }
```   
"""
class StartEndDurationStrategy(BaseDurationStrategy):
    def __init__(self, duration_config: Dict[str, Any]):
        super().__init__(type=DurationStrategyType.START_END_TIMESTAMP)
        self._start_column = duration_config[START_TIMESTAMP_PROP]
        self._end_column = duration_config[END_TIMESTAMP_PROP]
       
    def get_sort_by_column(self)-> str:
        return self._start_column      

    @property
    def start_column(self) -> str:
        return self._start_column
    
    @property
    def end_column(self) -> str:
        return self._end_column