# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import numpy as np
import pandas as pd

from models.base_model_factory import BaseModelFactory
from prep.config.prep_config import PrepConfig
from prep.data_prep import DataPrep
from forecasting.base_forecasting import BaseForecasting
from forecasting.config.forecasting_config import ForecastingConfig


class RemainingDurationForecasting(BaseForecasting):
    """Predict remaining time to case completion from a partial event prefix.

    Training data generation:
      For each complete case [E1, E2, ..., En] with cumulative durations [d1, d2, ..., dn],
      generate one sample per prefix length:
        prefix [E1]         -> target = dn - d1  (remaining from first event)
        prefix [E1, E2]     -> target = dn - d2  (remaining from second event)
        ...
        prefix [E1,..,En-1] -> target = dn - d(n-1)
    """

    def __init__(
            self,
            forecasting_config: ForecastingConfig,
            prep_config: PrepConfig,
            dataset: pd.DataFrame):
        super().__init__(forecasting_config, prep_config)
        self.event_columns = dataset.columns[
            dataset.columns.str.contains(forecasting_config.event_column_name_pattern.replace('*', '.*'))
        ]
        self.duration_columns = dataset.columns[
            dataset.columns.str.contains(forecasting_config.duration_column_name_pattern.replace('*', '.*'))
        ]
        print(f"Time series columns: {self.event_columns}")
        print(f"Duration columns: {self.duration_columns}")
        self.unique_events = None
        self.event_to_idx = None
        self.idx_to_event = None
        self._prep_config = prep_config
        self._dataset = dataset

    def get_forecasting_columns(self):
        return list(self.event_columns) + list(self.duration_columns)

    def preprocess_data(self, dataset: pd.DataFrame):
        sequences = dataset[self.event_columns].fillna('').values
        sequences = [list(filter(None, seq)) for seq in sequences if any(seq)]
        durations = dataset[self.duration_columns].fillna(0).values

        all_events = [event for seq in sequences for event in seq]
        self.unique_events = np.unique(all_events)
        self.event_to_idx = {event: idx for idx, event in enumerate(self.unique_events)}
        self.idx_to_event = {idx: event for event, idx in self.event_to_idx.items()}

        print(f"Unique events: {self.unique_events}")
        print(f"Event to index mapping: {self.event_to_idx}")

        def safe_map(event):
            return self.event_to_idx.get(event, -1)

        numerical_sequences = [[safe_map(event) for event in seq] for seq in sequences]

        valid_sequences = []
        valid_durations = []
        for seq, dur in zip(numerical_sequences, durations):
            if all(idx != -1 for idx in seq):
                valid_sequences.append(seq)
                valid_durations.append(dur)

        return valid_sequences, valid_durations

    def _generate_prefix_samples(self, event_sequences, duration_sequences):
        """Generate prefix-based training samples.

        For each complete case, create one sample per prefix length where
        the target is the remaining cumulative duration from that point to case end.
        """
        X_events = []
        X_durations = []
        y = []

        for seq, durs in zip(event_sequences, duration_sequences):
            n = len(seq)
            if n < 2:
                continue

            # durs are cumulative durations per event position
            durs_array = np.array(durs[:n], dtype='float32')
            case_end_duration = durs_array[-1]

            for prefix_len in range(1, n):
                event_prefix = seq[:prefix_len]
                duration_prefix = durs_array[:prefix_len].tolist()
                remaining = case_end_duration - durs_array[prefix_len - 1]
                X_events.append(event_prefix)
                X_durations.append(duration_prefix)
                y.append(float(remaining))

        return X_events, X_durations, np.array(y, dtype='float32')

    def train(self, model_factory: BaseModelFactory):
        prep_task = DataPrep(prep_config=self._prep_config, dataset=self._dataset.copy())
        preserved_columns = list(self.event_columns) + list(self.duration_columns)
        df = prep_task.prepare(preserved_columns=preserved_columns)

        valid_sequences, valid_durations = self.preprocess_data(df)
        print(f"Number of complete cases: {len(valid_sequences)}")

        X_events, X_durations, y = self._generate_prefix_samples(valid_sequences, valid_durations)
        print(f"Number of prefix training samples: {len(y)}")

        return model_factory.train_remaining_duration(
            event_sequences=X_events,
            duration_sequences=X_durations,
            unique_events_count=len(self.unique_events),
            y=y
        )

    def predict(self, model_factory: BaseModelFactory, X):
        in_progress_sequence = [self.event_to_idx.get(event, len(self.unique_events)) for event in X]
        return model_factory.predict_duration(in_progress_sequence)

    @staticmethod
    def compute_metrics(y_true, y_pred):
        """Compute MAE, MAPE, RMSE for duration predictions."""
        y_true = np.array(y_true, dtype='float32')
        y_pred = np.array(y_pred, dtype='float32')

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        nonzero = y_true != 0
        if nonzero.any():
            mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        else:
            mape = float('inf')

        return {"mae": mae, "mape": mape, "rmse": rmse}
