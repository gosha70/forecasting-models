# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from enum import Enum

class ModelType(Enum):
    LSTM = "LSTM"
    MC = "MarkovChain"
    RFC = "RandomForestClassifier"
    RFR = "RandomForestRegressor"
    TRANSFORMER = "Transformer"
    XGBOOST = "XGBoost"