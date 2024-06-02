# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import sys
import pandas as pd

from utils.file_util import load_json
from train.config.train_config import TrainConfig
from train.train_manager import TrainManager

def main(dataset_csv: str, config_json: str):
    config_dict = load_json(config_json)
    config = TrainConfig.from_dict(config_dict)
    dataset = pd.read_csv(dataset_csv, low_memory=False)
    train_manager = TrainManager(config, dataset)
    train_manager.execute()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m apps.build_model <Dataset CSV> <Forecasting Configuration JSON>")
    else:
        main(dataset_csv=sys.argv[1], config_json=sys.argv[2])
