# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import numpy as np
from typing import List, Dict, Optional

from models.base_model_factory import BaseModelFactory


class LSTMSimulator:
    """Autoregressive forward simulation using a trained LSTM model.

    Unlike the Markov-based ForwardSimulator which only uses the last event,
    this simulator feeds the full sequence history to the LSTM at each step,
    capturing higher-order dependencies between events.

    Flow per step:
      1. Call model.predict_proba(sequence_so_far) -> {event_idx: probability}
      2. Sample next event from that distribution
      3. Append to sequence, repeat until END or max_steps
    """

    def __init__(
            self,
            model_factory: BaseModelFactory,
            unique_events_count: int,
            max_steps: int = 100,
            seed: Optional[int] = None):
        self._model = model_factory
        self._unique_events_count = unique_events_count
        self._end_event_idx = unique_events_count  # END is mapped to unique_events_count
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed)

    def _sample_from_proba(self, proba: Dict) -> int:
        events = list(proba.keys())
        probs = np.array(list(proba.values()), dtype='float64')
        probs = probs / probs.sum()
        return self._rng.choice(events, p=probs)

    def next_event_distribution(self, sequence: List[int]) -> Dict:
        """Return {event_idx: probability} for the next step given full sequence."""
        return self._model.predict_proba(sequence)

    def simulate_trajectory(self, start_sequence: List[int]) -> List[int]:
        """Simulate from a starting sequence until END event or max_steps."""
        trajectory = list(start_sequence)
        for _ in range(self._max_steps):
            proba = self._model.predict_proba(trajectory)
            next_event = self._sample_from_proba(proba)
            if next_event == self._end_event_idx:
                break
            trajectory.append(next_event)
        return trajectory

    def simulate_batch(self, start_sequence: List[int], n_simulations: int) -> List[List[int]]:
        """Generate n_simulations trajectories from the same starting sequence."""
        return [self.simulate_trajectory(start_sequence) for _ in range(n_simulations)]

    def simulate_from_prefix(self, prefix: List[int], n_simulations: int = 1) -> List[List[int]]:
        """Continue simulation from an in-progress case prefix."""
        if not prefix:
            raise ValueError("prefix must contain at least one event")
        return [self.simulate_trajectory(prefix) for _ in range(n_simulations)]

    def trajectory_statistics(self, trajectories: List[List[int]]) -> Dict:
        """Compute basic statistics over a batch of simulated trajectories."""
        lengths = [len(t) for t in trajectories]
        end_events = [t[-1] for t in trajectories]

        end_event_counts = {}
        for e in end_events:
            end_event_counts[e] = end_event_counts.get(e, 0) + 1

        total = len(trajectories)
        end_event_probs = {e: c / total for e, c in end_event_counts.items()}

        return {
            "count": total,
            "mean_length": float(np.mean(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "std_length": float(np.std(lengths)),
            "end_event_distribution": end_event_probs,
        }
