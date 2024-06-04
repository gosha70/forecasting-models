
# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from typing import List, Dict, Any, Tuple

TEST_DATA_PROP = 'test_data'

class TestData:
    def __init__(self, test_data: Dict[str, Any]):
        self._test_data = test_data
    
    @property
    def test_data(self):
        return self._test_data

    def get_X_y_pairs(self)->List[Tuple[List[str], str]]:
        return [(item["X"], item["y"]) for item in self.test_data]
