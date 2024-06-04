
# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import Any, Dict

from .base_duration_strategy import BaseDurationStrategy
from .transformation_type import TransformationType
from .merge_strategy_type import MergeStrategyType

TRANSFORMATION_TYPE_PROP = 'transformation_type'
CASE_ID_COLUMN_PROP = 'case_id_column'
MIN_EVENT_SEQUENCE = 'min_event_sequence'
MAX_EVENT_SEQUENCE = 'max_event_sequence'
EVENT_COLUMN_PROP = 'event_column'
MERGE_STRATEGY_PROP = 'merge_strategy'
EVENT_PREFIX_PROP = 'event_prexix'
EVENT_DURATION_PREFIX_PROP = 'event_duration_prexix'

"""
```
{
   "transformation_type": 1,
   "case_id_column": "number",
   "event_column": "incident_state",
   "min_event_sequence": 2,
   "max_event_sequence": 20,
   "duration_strategy": {
      "type": 2,
      "time_column": "sys_updated_at"
   },
   "merge_strategy": 3,
   "event_prexix": "__EVENT_",
   "event_duration_prexix": "__DURATION_EVENT_"
}
```
"""
class TransformationConfig:
    def __init__(
        self, 
        transformation_type: TransformationType, 
        case_id_column: str, 
        event_column: str, 
        min_events: int,
        max_events: int,
        duration_strategy: BaseDurationStrategy, 
        merge_strategy_type: MergeStrategyType,
        event_prefix: str,
        event_duration_prefix: str
    ):
        self._transformation_type = transformation_type
        self._case_id_column = case_id_column
        self._event_column = event_column
        self._min_events = min_events
        self._max_events = max_events
        self._duration_strategy = duration_strategy
        self._merge_strategy_type = merge_strategy_type
        self._event_prefix = event_prefix
        self._event_duration_prefix = event_duration_prefix
    
    @property
    def transformation_type(self) -> TransformationType:
        return self._transformation_type   
        
    @property
    def case_id_column(self) -> str:
        return self._case_id_column  
        
    @property
    def event_column(self) -> str:
        return self._event_column    
        
    @property
    def min_events(self) -> str:
        return self._min_events    
        
    @property
    def max_events(self) -> str:
        return self._max_events    
        
    @property
    def duration_strategy(self) -> BaseDurationStrategy:
        return self._duration_strategy   
        
    @property
    def merge_strategy_type(self) -> MergeStrategyType:
        return self._merge_strategy_type
    
    @property
    def event_prefix(self) -> str:
        return self._event_prefix  
        
    @property
    def event_duration_prefix(self) -> str:
        return self._event_duration_prefix    

    @classmethod
    def from_dict(cls, transformation_config: Dict[str, Any]):
        return cls(
            transformation_type=TransformationType(transformation_config[TRANSFORMATION_TYPE_PROP]),
            case_id_column=transformation_config[CASE_ID_COLUMN_PROP],
            event_column=transformation_config[EVENT_COLUMN_PROP],
            min_events = int(transformation_config[MIN_EVENT_SEQUENCE]),
            max_events = int(transformation_config[MAX_EVENT_SEQUENCE]), 
            duration_strategy=BaseDurationStrategy.create_duration_strategy(transformation_config),
            merge_strategy_type=MergeStrategyType(transformation_config[MERGE_STRATEGY_PROP]),
            event_prefix=transformation_config[EVENT_PREFIX_PROP],
            event_duration_prefix=transformation_config[EVENT_DURATION_PREFIX_PROP]
        )