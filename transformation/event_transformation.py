# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import pandas as pd

from .transformation_config import TransformationConfig
from .merge_strategy_type import MergeStrategyType

class EventTransformation:
    def __init__(
            self, 
            transformation_config: TransformationConfig, 
            dataset: pd.DataFrame):
        self._config = transformation_config
        self._df = dataset

    @property
    def config(self) -> TransformationConfig:
        return self._config    
    
    @property
    def dataset(self) -> pd.DataFrame:
        return self._df    

    def execute(self) -> pd.DataFrame:  
        case_id_column = self.config.case_id_column
        event_column = self.config.event_column
        event_prefix = self.config.event_prefix
        event_duration_prefix = self.config.event_duration_prefix
        duration_strategy = self.config.duration_strategy
        sort_by_column = duration_strategy.get_sort_by_column()
        
        # Convert to datetime
        self.dataset[sort_by_column] = pd.to_datetime(self.dataset[sort_by_column], dayfirst=True)

        # Sort the dataset by the case id and time stamp
        df_sorted = self.dataset.sort_values(by=[case_id_column, sort_by_column])

        # Group by the case id  and aggregate events in the sorted order
        new_dataset = df_sorted.groupby(case_id_column).agg(list)

        # Filter out cases with number of events < min_events or > max_events
        new_dataset = new_dataset[(new_dataset[event_column].apply(len) >= self.config.min_events) & (new_dataset[event_column].apply(len) <= self.config.max_events)]
        
        # Create new columns for each state in the sequence
        max_length = new_dataset[event_column].apply(len).max()
        for i in range(max_length):
            new_dataset[f'{event_prefix}{i+1}'] = new_dataset[event_column].apply(lambda x: x[i] if i < len(x) else '')
        
        # Calculate duration for each event
        new_dataset = duration_strategy.calculate_duration(dataset=new_dataset, max_length=max_length, name_prefix=event_duration_prefix)

        # Drop the original event column
        new_dataset = new_dataset.drop(columns=[event_column])

        # Reset index to convert case from index to column
        new_dataset = new_dataset.reset_index()

        # Apply merge strategy
        return self.apply_merge_strategy(dataset=new_dataset)
    

    def apply_merge_strategy(self, dataset: pd.DataFrame) -> pd.DataFrame:
        merge_strategy = self.config.merge_strategy_type
        # Apply cleanup strategy
        if merge_strategy == MergeStrategyType.SEQUENCE:
            return dataset
        elif merge_strategy == MergeStrategyType.LAST:
            for col in dataset.columns:
                if isinstance(dataset[col].iloc[0], list):
                    dataset[col] = dataset[col].apply(lambda x: x[-1])
        elif merge_strategy == MergeStrategyType.AVERAGE:
            for col in dataset.columns:
                if isinstance(dataset[col].iloc[0], list):
                    if pd.api.types.is_numeric_dtype(dataset[col].apply(lambda x: x[0])):
                        dataset[col] = dataset[col].apply(lambda x: sum(x) / len(x) if x else None)
                    else:
                        dataset[col] = dataset[col].apply(lambda x: max(set(x), key=x.count) if x else None)
        
        return dataset
