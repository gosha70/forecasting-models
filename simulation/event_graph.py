# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import networkx as nx


class EventGraph:
    """Directed graph representation of a process event transition model.

    Nodes are events, edges carry transition probabilities.
    Built from a Markov Chain transition matrix (dict-of-dicts).
    """

    def __init__(self, transition_matrix: dict):
        self._transition_matrix = transition_matrix
        self._graph = nx.DiGraph()
        self._build_graph()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def transition_matrix(self) -> dict:
        return self._transition_matrix

    def _build_graph(self):
        for source, targets in self._transition_matrix.items():
            for target, prob in targets.items():
                self._graph.add_edge(source, target, weight=prob)

    @property
    def events(self):
        return list(self._graph.nodes)

    @property
    def start_events(self):
        """Events with no incoming edges (process entry points)."""
        return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

    @property
    def end_events(self):
        """Absorbing events — no outgoing transitions."""
        return [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]

    @property
    def transient_events(self):
        """Non-absorbing events — have at least one outgoing transition."""
        return [n for n in self._graph.nodes if self._graph.out_degree(n) > 0]

    def get_successors(self, event):
        """Return {successor_event: probability} for the given event."""
        return self._transition_matrix.get(event, {})

    def get_predecessors(self, event):
        """Return list of events that can transition into the given event."""
        return list(self._graph.predecessors(event))
