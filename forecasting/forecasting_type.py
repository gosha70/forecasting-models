# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
from enum import Enum

class ForecastingType(Enum):
    NEXT_EVENT = "next_event"
    EVENT_DURATION = "event_duration"
    CASE_DURATION = "case_duration"
    CASE_CLASS = "case_class"