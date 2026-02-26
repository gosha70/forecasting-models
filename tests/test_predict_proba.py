# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import pytest
import numpy as np

from models.markov_chain_factory import MarkivChainModelFactory
from models.base_model_factory import BaseModelFactory


class TestMarkovChainPredictProba:
    def setup_method(self):
        self.mc = MarkivChainModelFactory()
        self.mc.transition_matrix = {
            0: {1: 0.7, 2: 0.3},
            1: {2: 0.6, 3: 0.4},
            2: {3: 1.0},
        }
        self.mc.unique_events_count = 4

    def test_returns_dict(self):
        result = self.mc.predict_proba([0])
        assert isinstance(result, dict)

    def test_probabilities_sum_to_one(self):
        result = self.mc.predict_proba([0])
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_correct_distribution(self):
        result = self.mc.predict_proba([0])
        assert abs(result[1] - 0.7) < 1e-6
        assert abs(result[2] - 0.3) < 1e-6

    def test_uses_last_event_in_sequence(self):
        result = self.mc.predict_proba([0, 1])
        assert abs(result[2] - 0.6) < 1e-6
        assert abs(result[3] - 0.4) < 1e-6

    def test_absorbing_state_returns_end(self):
        result = self.mc.predict_proba([3])
        assert result == {4: 1.0}  # unique_events_count = END index

    def test_deterministic_transition(self):
        result = self.mc.predict_proba([2])
        assert result == {3: 1.0}

    def test_predict_matches_argmax_of_proba(self):
        for seq in [[0], [1], [0, 1], [0, 1, 2]]:
            predicted = self.mc.predict(seq)
            proba = self.mc.predict_proba(seq)
            if proba:
                best_from_proba = max(proba, key=proba.get)
                assert predicted == best_from_proba


class TestNextEventForecastingPredictProba:
    """Test predict_proba at the forecasting layer with event name mapping."""

    def setup_method(self):
        self.mc = MarkivChainModelFactory()
        self.mc.transition_matrix = {
            0: {1: 0.7, 2: 0.3},
            1: {2: 0.6, 3: 0.4},
            2: {3: 1.0},
        }
        self.mc.unique_events_count = 4

    def _make_forecasting(self):
        """Create a minimal NextEventForecasting with preset mappings."""
        from forecasting.next_event_forecasting import NextEventForecasting
        from forecasting.config.forecasting_config import ForecastingConfig
        from forecasting.forecasting_type import ForecastingType
        from prep.config.prep_config import PrepConfig
        import pandas as pd

        events = ["New", "Accepted", "Billing", "Delivered"]
        event_to_idx = {e: i for i, e in enumerate(events)}
        idx_to_event = {i: e for i, e in enumerate(events)}

        df = pd.DataFrame({"__EVENT_1": ["New"], "__EVENT_2": ["Accepted"]})
        fc = ForecastingConfig(
            forecasting_type=ForecastingType.NEXT_EVENT,
            name_pattern="__EVENT_*",
            duration_pattern=None,
            is_random_seq=False,
        )
        pc = PrepConfig(include_all_data=False, drop_colum_names=[], drop_colum_patterns=[])

        nef = NextEventForecasting(fc, pc, df)
        nef.unique_events = np.array(events)
        nef.event_to_idx = event_to_idx
        nef.idx_to_event = idx_to_event
        return nef

    def test_returns_named_events(self):
        nef = self._make_forecasting()
        result = nef.predict_proba(self.mc, ["New"])
        assert "Accepted" in result
        assert "Billing" in result
        assert len(result) == 2

    def test_probabilities_sum_to_one(self):
        nef = self._make_forecasting()
        result = nef.predict_proba(self.mc, ["New"])
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_sorted_descending(self):
        nef = self._make_forecasting()
        result = nef.predict_proba(self.mc, ["New"])
        probs = list(result.values())
        assert probs == sorted(probs, reverse=True)

    def test_top_k(self):
        nef = self._make_forecasting()
        result = nef.predict_proba(self.mc, ["New"], top_k=1)
        assert len(result) == 1
        assert "Accepted" in result

    def test_threshold(self):
        nef = self._make_forecasting()
        result = nef.predict_proba(self.mc, ["New"], threshold=0.5)
        assert "Accepted" in result
        assert "Billing" not in result


class TestBaseModelFactoryPredictProba:
    def test_predict_proba_is_abstract(self):
        assert hasattr(BaseModelFactory, 'predict_proba')
