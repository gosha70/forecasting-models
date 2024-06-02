# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import Dict
import json

def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)
