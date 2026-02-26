# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import pytest
import numpy as np
import pandas as pd

from forecasting.forecasting_type import ForecastingType
from forecasting.base_forecasting import BaseForecasting
from forecasting.config.forecasting_config import ForecastingConfig
from prep.config.prep_config import PrepConfig


class TestForecastingTypeEnum:
    def test_remaining_duration_exists(self):
        assert hasattr(ForecastingType, 'REMAINING_DURATION')

    def test_remaining_duration_value(self):
        assert ForecastingType.REMAINING_DURATION.value == "remaining_duration"

    def test_all_types_present(self):
        expected = {"next_event", "event_duration", "case_duration", "case_class", "remaining_duration"}
        actual = {t.value for t in ForecastingType}
        assert actual == expected


class TestBaseForcastingFactory:
    def _make_config(self, forecasting_type_str):
        fc = ForecastingConfig(
            forecasting_type=ForecastingType[forecasting_type_str.upper()],
            name_pattern="__EVENT_*",
            duration_pattern="__DURATION_EVENT_*",
            is_random_seq=False,
        )
        pc = PrepConfig(include_all_data=False, drop_colum_names=[], drop_colum_patterns=[])
        return fc, pc

    def _make_dataset(self):
        return pd.DataFrame({
            "__EVENT_1": ["A", "A"],
            "__EVENT_2": ["B", "B"],
            "__DURATION_EVENT_1": [1.0, 2.0],
            "__DURATION_EVENT_2": [3.0, 4.0],
        })

    def test_creates_remaining_duration_forecasting(self):
        from forecasting.remaining_duration_forecasting import RemainingDurationForecasting
        fc, pc = self._make_config("remaining_duration")
        ds = self._make_dataset()
        result = BaseForecasting.create_prediction_training(fc, pc, ds)
        assert isinstance(result, RemainingDurationForecasting)

    def test_creates_next_event_forecasting(self):
        from forecasting.next_event_forecasting import NextEventForecasting
        fc, pc = self._make_config("next_event")
        ds = self._make_dataset()
        result = BaseForecasting.create_prediction_training(fc, pc, ds)
        assert isinstance(result, NextEventForecasting)


class TestRemainingDurationPrefixGeneration:
    def test_generates_correct_number_of_samples(self):
        from forecasting.remaining_duration_forecasting import RemainingDurationForecasting

        fc = ForecastingConfig(
            forecasting_type=ForecastingType.REMAINING_DURATION,
            name_pattern="__EVENT_*",
            duration_pattern="__DURATION_EVENT_*",
            is_random_seq=False,
        )
        pc = PrepConfig(include_all_data=False, drop_colum_names=[], drop_colum_patterns=[])
        ds = pd.DataFrame({
            "__EVENT_1": ["A", "A", "A"],
            "__EVENT_2": ["B", "B", "B"],
            "__EVENT_3": ["C", "C", ""],
            "__DURATION_EVENT_1": [1.0, 1.0, 1.0],
            "__DURATION_EVENT_2": [3.0, 3.0, 3.0],
            "__DURATION_EVENT_3": [6.0, 6.0, 0.0],
        })

        rdf = RemainingDurationForecasting(fc, pc, ds)

        # Call internal method directly
        # Case [A,B,C] with durations [1,3,6] -> 2 prefixes (len 1 and len 2)
        # Case [A,B] with durations [1,3] -> 1 prefix (len 1)
        event_seqs = [[0, 1, 2], [0, 1, 2], [0, 1]]
        dur_seqs = [[1.0, 3.0, 6.0], [1.0, 3.0, 6.0], [1.0, 3.0]]
        X_events, X_durations, y = rdf._generate_prefix_samples(event_seqs, dur_seqs)

        # 3-event cases generate 2 prefixes each, 2-event case generates 1 prefix
        assert len(y) == 5  # 2 + 2 + 1

    def test_prefix_targets_are_correct(self):
        from forecasting.remaining_duration_forecasting import RemainingDurationForecasting

        fc = ForecastingConfig(
            forecasting_type=ForecastingType.REMAINING_DURATION,
            name_pattern="__EVENT_*",
            duration_pattern="__DURATION_EVENT_*",
            is_random_seq=False,
        )
        pc = PrepConfig(include_all_data=False, drop_colum_names=[], drop_colum_patterns=[])
        ds = pd.DataFrame({
            "__EVENT_1": ["A"],
            "__EVENT_2": ["B"],
            "__EVENT_3": ["C"],
            "__DURATION_EVENT_1": [2.0],
            "__DURATION_EVENT_2": [5.0],
            "__DURATION_EVENT_3": [10.0],
        })

        rdf = RemainingDurationForecasting(fc, pc, ds)

        # Case [A,B,C] durations [2, 5, 10]
        # Prefix [A] -> remaining = 10 - 2 = 8
        # Prefix [A,B] -> remaining = 10 - 5 = 5
        event_seqs = [[0, 1, 2]]
        dur_seqs = [[2.0, 5.0, 10.0]]
        X_events, X_durations, y = rdf._generate_prefix_samples(event_seqs, dur_seqs)

        assert len(y) == 2
        assert abs(y[0] - 8.0) < 1e-6  # [A] -> 10-2
        assert abs(y[1] - 5.0) < 1e-6  # [A,B] -> 10-5

    def test_single_event_case_skipped(self):
        from forecasting.remaining_duration_forecasting import RemainingDurationForecasting

        fc = ForecastingConfig(
            forecasting_type=ForecastingType.REMAINING_DURATION,
            name_pattern="__EVENT_*",
            duration_pattern="__DURATION_EVENT_*",
            is_random_seq=False,
        )
        pc = PrepConfig(include_all_data=False, drop_colum_names=[], drop_colum_patterns=[])
        ds = pd.DataFrame({
            "__EVENT_1": ["A"],
            "__DURATION_EVENT_1": [1.0],
        })

        rdf = RemainingDurationForecasting(fc, pc, ds)
        X_events, X_durations, y = rdf._generate_prefix_samples([[0]], [[1.0]])
        assert len(y) == 0


class TestComputeMetrics:
    def test_perfect_predictions(self):
        from forecasting.remaining_duration_forecasting import RemainingDurationForecasting
        metrics = RemainingDurationForecasting.compute_metrics([10, 20, 30], [10, 20, 30])
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mape"] == 0.0

    def test_known_error(self):
        from forecasting.remaining_duration_forecasting import RemainingDurationForecasting
        metrics = RemainingDurationForecasting.compute_metrics([10, 20], [12, 18])
        assert abs(metrics["mae"] - 2.0) < 1e-4
        assert abs(metrics["rmse"] - 2.0) < 1e-4
        # MAPE: (|2/10| + |2/20|) / 2 * 100 = (0.2 + 0.1) / 2 * 100 = 15
        assert abs(metrics["mape"] - 15.0) < 1e-2
