# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import List, Dict, Any

DROP_COLUMN_NAMES_PROP = 'drop_colum_names'
DROP_COLUMN_PATTERNS_PROP = 'drop_colum_patterns'
INCLUDE_ALL_DATA_PROP = 'include_all_data'

class PrepConfig:
    def __init__(
            self, 
            include_all_data: bool,
            drop_colum_names: List[str], 
            drop_colum_patterns: List[str]):
        self._include_all_data = include_all_data
        self._drop_colum_names = drop_colum_names
        self._drop_colum_patterns = drop_colum_patterns
                            
    @property
    def include_all_data(self)-> bool:
        return self._include_all_data     
           
    @property
    def drop_colum_names(self)-> str:
        return self._drop_colum_names        
                 
    @property
    def drop_colum_patterns(self)-> str:
        return self._drop_colum_patterns    

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        drop_colum_names = data[DROP_COLUMN_NAMES_PROP]
        drop_colum_patterns = data[DROP_COLUMN_PATTERNS_PROP]
        include_all_data = data[INCLUDE_ALL_DATA_PROP] 
        return cls(include_all_data, drop_colum_names, drop_colum_patterns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            DROP_COLUMN_NAMES_PROP: self.drop_colum_names,
            DROP_COLUMN_PATTERNS_PROP: self.drop_colum_patterns,
            INCLUDE_ALL_DATA_PROP: self.include_all_data
        }