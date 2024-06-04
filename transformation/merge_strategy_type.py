# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from enum import Enum

class MergeStrategyType(Enum):
    SEQUENCE = 1
    AVERAGE= 2
    LAST = 3