# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import numpy as np
from typing import List, Dict, Optional

from .event_graph import EventGraph


class ForwardSimulator:
    """Monte Carlo forward simulation over a process event graph.

    Supports:
    - Single-step: probability distribution of next event from a given state
    - Multi-step: sample a trajectory of N steps from a starting event
    - Full trajectory: sample until an absorbing (end) event is reached
    - Batch: generate K complete case trajectories for statistical analysis
    """

    def __init__(self, event_graph: EventGraph, max_steps: int = 100, seed: Optional[int] = None):
        self._graph = event_graph
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed)

    @property
    def event_graph(self) -> EventGraph:
        return self._graph

    def next_event_distribution(self, current_event) -> Dict:
        """Return {event: probability} for a single step from current_event."""
        return self._graph.get_successors(current_event)

    def sample_next_event(self, current_event):
        """Sample one next event according to transition probabilities.

        Returns None if current_event is absorbing.
        """
        successors = self._graph.get_successors(current_event)
        if not successors:
            return None
        events = list(successors.keys())
        probs = list(successors.values())
        return self._rng.choice(events, p=probs)

    def simulate_steps(self, start_event, n_steps: int) -> List:
        """Simulate exactly n_steps from start_event.

        Returns a trajectory list including start_event.
        Stops early if an absorbing state is reached.
        """
        trajectory = [start_event]
        current = start_event
        for _ in range(n_steps):
            next_ev = self.sample_next_event(current)
            if next_ev is None:
                break
            trajectory.append(next_ev)
            current = next_ev
        return trajectory

    def simulate_trajectory(self, start_event) -> List:
        """Simulate a full trajectory from start_event until an end event or max_steps."""
        return self.simulate_steps(start_event, self._max_steps)

    def simulate_batch(self, start_event, n_simulations: int) -> List[List]:
        """Generate n_simulations complete trajectories from start_event."""
        return [self.simulate_trajectory(start_event) for _ in range(n_simulations)]

    def simulate_from_prefix(self, prefix: List, n_simulations: int = 1) -> List[List]:
        """Continue simulation from an in-progress case prefix.

        Each result is the full sequence: prefix + simulated continuation.
        """
        if not prefix:
            raise ValueError("prefix must contain at least one event")
        results = []
        last_event = prefix[-1]
        for _ in range(n_simulations):
            continuation = self.simulate_trajectory(last_event)
            results.append(list(prefix) + continuation[1:])  # avoid duplicating last_event
        return results

    def trajectory_statistics(self, trajectories: List[List]) -> Dict:
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
