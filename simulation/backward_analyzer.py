# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from .event_graph import EventGraph


class BackwardAnalyzer:
    """Backward analysis of a process event graph.

    Provides:
    - Reverse transition matrix: P(predecessor | current_event)
    - Absorption probabilities: from transient state i, probability of eventually
      reaching each absorbing state j.

    Uses standard Markov chain absorbing state analysis:
      1. Partition states into transient (T) and absorbing (A)
      2. Extract sub-matrix Q (transient → transient) and R (transient → absorbing)
      3. Fundamental matrix N = (I - Q)^(-1)
      4. Absorption matrix B = N × R
      5. B[i][j] = P(eventually absorbed into state j | start in transient state i)
    """

    def __init__(self, event_graph: EventGraph):
        self._graph = event_graph
        self._reverse_matrix = None
        self._absorption_matrix = None
        self._transient_states = None
        self._absorbing_states = None
        self._transient_idx = None
        self._absorbing_idx = None

        self._build_reverse_matrix()
        self._compute_absorption_probabilities()

    @property
    def reverse_transition_matrix(self) -> Dict:
        return self._reverse_matrix

    @property
    def absorbing_states(self) -> List:
        return self._absorbing_states

    @property
    def transient_states(self) -> List:
        return self._transient_states

    def _build_reverse_matrix(self):
        """Compute P(predecessor | current_event) from the forward transition counts."""
        incoming = defaultdict(lambda: defaultdict(float))
        for source, targets in self._graph.transition_matrix.items():
            for target, prob in targets.items():
                incoming[target][source] += prob

        self._reverse_matrix = {}
        for event, predecessors in incoming.items():
            total = sum(predecessors.values())
            if total > 0:
                self._reverse_matrix[event] = {
                    pred: weight / total for pred, weight in predecessors.items()
                }

    def _compute_absorption_probabilities(self):
        """Compute the absorption probability matrix B via fundamental matrix N."""
        self._transient_states = self._graph.transient_events
        self._absorbing_states = self._graph.end_events

        if not self._absorbing_states or not self._transient_states:
            self._absorption_matrix = None
            return

        self._transient_idx = {s: i for i, s in enumerate(self._transient_states)}
        self._absorbing_idx = {s: i for i, s in enumerate(self._absorbing_states)}

        n_t = len(self._transient_states)
        n_a = len(self._absorbing_states)

        Q = np.zeros((n_t, n_t))
        R = np.zeros((n_t, n_a))

        for source, targets in self._graph.transition_matrix.items():
            if source not in self._transient_idx:
                continue
            i = self._transient_idx[source]
            for target, prob in targets.items():
                if target in self._transient_idx:
                    Q[i][self._transient_idx[target]] = prob
                elif target in self._absorbing_idx:
                    R[i][self._absorbing_idx[target]] = prob

        # N = (I - Q)^(-1)
        I = np.eye(n_t)
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            self._absorption_matrix = None
            return

        # B = N × R
        self._absorption_matrix = N @ R

    def get_absorption_probability(self, from_event, to_event) -> Optional[float]:
        """P(eventually reaching absorbing to_event | starting at transient from_event)."""
        if self._absorption_matrix is None:
            return None
        if from_event not in self._transient_idx or to_event not in self._absorbing_idx:
            return None
        i = self._transient_idx[from_event]
        j = self._absorbing_idx[to_event]
        return float(self._absorption_matrix[i][j])

    def get_all_absorption_probabilities(self, from_event) -> Optional[Dict]:
        """Return {absorbing_event: probability} for all absorbing states from from_event."""
        if self._absorption_matrix is None:
            return None
        if from_event not in self._transient_idx:
            return None
        i = self._transient_idx[from_event]
        return {
            state: float(self._absorption_matrix[i][j])
            for state, j in self._absorbing_idx.items()
        }

    def get_reverse_predecessors(self, event) -> Dict:
        """Return {predecessor: P(predecessor | event)} from the reverse matrix."""
        return self._reverse_matrix.get(event, {})
