# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import sys
import pandas as pd

from utils.file_util import load_json
from transformation.transformation_config import TransformationConfig
from transformation.transformation_manager import TransformationManager

EVENT_CASE_OUTPUT_CSV = "event_case_output.csv"

def main(config_json: str, dataset1_csv: str, dataset2_csv: str):
    config_dict = load_json(config_json)
    print(f"Configuration: {config_dict}")
    config = TransformationConfig.from_dict(transformation_config=config_dict)
    dataset1 = pd.read_csv(dataset1_csv, low_memory=False)
    if dataset2_csv:
        dataset2 = pd.read_csv(dataset2_csv, low_memory=False)
    else:
        dataset2 = None
    transformation_config = TransformationManager(config, dataset1, dataset2)
    to_dataset = transformation_config.execute()
    print(f"Output dataset shape: {to_dataset.shape}")
    print(f"Creating CSV with the transformed dataset: {EVENT_CASE_OUTPUT_CSV} ...")
    to_dataset.to_csv(EVENT_CASE_OUTPUT_CSV, index=False)

if __name__ == "__main__":
    args_len = len(sys.argv)
    if args_len < 3:
        print("Usage: python -m apps.transform <Transformation Configuration JSON> <Dataset #1 CSV>  <Dataset #2 CSV> ")
    else:
        if args_len == 4:
            main(config_json=sys.argv[1], dataset1_csv=sys.argv[2], dataset2_csv=sys.argv[3])
        else:
            main(config_json=sys.argv[1], dataset1_csv=sys.argv[2], dataset2_csv=None)
       
