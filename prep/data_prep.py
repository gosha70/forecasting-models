# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import List
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from prep.config.prep_config import PrepConfig

class DataPrep:
    """The class for preparing dataset right before creating a training model"""
  
    def __init__(self, prep_config: PrepConfig, dataset: pd.DataFrame):
        """Initializes BaseFileConverter with optional Language and Logging."""
        self._config = prep_config
        self.df = dataset
        self._scalers = {}

    @property
    def scalers(self):
        return self._scalers
    
    def prepare(self, preserved_columns: List[str]) -> pd.DataFrame:
        if self._config.include_all_data:
            self.drop_columns()   
            self.impute_columns(preserved_columns)
        else:   
            # Retain only the specified columns
            self.df = self.df.loc[:, preserved_columns]
        return self.df
    
    def drop_columns(self):
        # Drop specified columns
        self.df.drop(columns=self._config.drop_colum_names, inplace=True, errors='ignore')

        # Drop columns based on patterns
        for pattern in self._config.drop_colum_patterns:
            columns_to_drop = self.df.columns[self.df.columns.str.contains(pattern.replace('*', '.*'))]
            self.df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    def impute_columns(self, preserved_columns: List[str]): 
        # Identify additional features (columns not starting with the event prefix)
        impute_columns = [col for col in self.df.columns if col not in preserved_columns]

        # Handle missing values for additional features
        self.df[impute_columns] = self.df[impute_columns].ffill().bfill()
        
        # Encode categorical features (excluding event columns)
        for col in self.df[impute_columns].select_dtypes(include=['object']).columns:
            self.df[col] = LabelEncoder().fit_transform(self.df[col])
        
        # Normalize numerical features (excluding event columns)
        for col in self.df[impute_columns].select_dtypes(include=['float64', 'int64']).columns:
            scaler = StandardScaler()
            self.df[col] = scaler.fit_transform(self.df[col].values.reshape(-1, 1))
            self._scalers[col] = scaler