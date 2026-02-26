# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import pytest
import numpy as np

from simulation.event_graph import EventGraph
from simulation.forward_simulator import ForwardSimulator
from simulation.backward_analyzer import BackwardAnalyzer


# Shared transition matrix: 0 -> 1|2, 1 -> 2|3, 2 -> 3 (absorbing: 3)
TRANSITION_MATRIX = {
    0: {1: 0.7, 2: 0.3},
    1: {2: 0.6, 3: 0.4},
    2: {3: 1.0},
}


class TestEventGraph:
    def setup_method(self):
        self.graph = EventGraph(TRANSITION_MATRIX)

    def test_events_contains_all_nodes(self):
        assert set(self.graph.events) == {0, 1, 2, 3}

    def test_start_events(self):
        assert self.graph.start_events == [0]

    def test_end_events(self):
        assert self.graph.end_events == [3]

    def test_transient_events(self):
        assert set(self.graph.transient_events) == {0, 1, 2}

    def test_get_successors(self):
        assert self.graph.get_successors(0) == {1: 0.7, 2: 0.3}

    def test_get_successors_absorbing(self):
        assert self.graph.get_successors(3) == {}

    def test_get_predecessors(self):
        preds = self.graph.get_predecessors(2)
        assert set(preds) == {0, 1}

    def test_get_predecessors_start(self):
        assert self.graph.get_predecessors(0) == []

    def test_graph_edge_weights(self):
        g = self.graph.graph
        assert g[0][1]["weight"] == 0.7
        assert g[0][2]["weight"] == 0.3


class TestForwardSimulator:
    def setup_method(self):
        self.graph = EventGraph(TRANSITION_MATRIX)
        self.sim = ForwardSimulator(self.graph, max_steps=50, seed=42)

    def test_next_event_distribution(self):
        dist = self.sim.next_event_distribution(0)
        assert dist == {1: 0.7, 2: 0.3}

    def test_sample_next_event_valid(self):
        event = self.sim.sample_next_event(0)
        assert event in [1, 2]

    def test_sample_next_event_absorbing(self):
        assert self.sim.sample_next_event(3) is None

    def test_simulate_trajectory_starts_with_start(self):
        traj = self.sim.simulate_trajectory(0)
        assert traj[0] == 0

    def test_simulate_trajectory_ends_at_absorbing(self):
        traj = self.sim.simulate_trajectory(0)
        assert traj[-1] == 3

    def test_simulate_trajectory_valid_transitions(self):
        traj = self.sim.simulate_trajectory(0)
        for i in range(len(traj) - 1):
            successors = self.graph.get_successors(traj[i])
            assert traj[i + 1] in successors

    def test_simulate_batch_count(self):
        batch = self.sim.simulate_batch(0, 50)
        assert len(batch) == 50

    def test_simulate_batch_all_end_at_absorbing(self):
        batch = self.sim.simulate_batch(0, 100)
        for traj in batch:
            assert traj[-1] == 3

    def test_simulate_from_prefix(self):
        results = self.sim.simulate_from_prefix([0, 1], n_simulations=10)
        for traj in results:
            assert traj[0] == 0
            assert traj[1] == 1
            assert traj[-1] == 3

    def test_simulate_from_prefix_empty_raises(self):
        with pytest.raises(ValueError):
            self.sim.simulate_from_prefix([], n_simulations=1)

    def test_trajectory_statistics_count(self):
        batch = self.sim.simulate_batch(0, 100)
        stats = self.sim.trajectory_statistics(batch)
        assert stats["count"] == 100

    def test_trajectory_statistics_end_distribution_sums_to_one(self):
        batch = self.sim.simulate_batch(0, 1000)
        stats = self.sim.trajectory_statistics(batch)
        total = sum(stats["end_event_distribution"].values())
        assert abs(total - 1.0) < 1e-6

    def test_simulate_steps_respects_n(self):
        traj = self.sim.simulate_steps(0, 1)
        assert len(traj) <= 2  # start + at most 1 step

    def test_deterministic_seed(self):
        sim1 = ForwardSimulator(self.graph, seed=123)
        sim2 = ForwardSimulator(self.graph, seed=123)
        t1 = sim1.simulate_batch(0, 20)
        t2 = sim2.simulate_batch(0, 20)
        assert t1 == t2


class TestBackwardAnalyzer:
    def setup_method(self):
        self.graph = EventGraph(TRANSITION_MATRIX)
        self.ba = BackwardAnalyzer(self.graph)

    def test_absorbing_states(self):
        assert self.ba.absorbing_states == [3]

    def test_transient_states(self):
        assert set(self.ba.transient_states) == {0, 1, 2}

    def test_absorption_probabilities_sum_to_one(self):
        for t in self.ba.transient_states:
            probs = self.ba.get_all_absorption_probabilities(t)
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-6, f"From {t}: sum={total}"

    def test_absorption_from_state_2(self):
        # State 2 -> 3 with prob 1.0
        prob = self.ba.get_absorption_probability(2, 3)
        assert abs(prob - 1.0) < 1e-6

    def test_absorption_from_state_0(self):
        # All paths from 0 eventually reach 3
        prob = self.ba.get_absorption_probability(0, 3)
        assert abs(prob - 1.0) < 1e-6

    def test_absorption_unknown_from_event(self):
        assert self.ba.get_absorption_probability(99, 3) is None

    def test_absorption_unknown_to_event(self):
        assert self.ba.get_absorption_probability(0, 99) is None

    def test_reverse_matrix_exists(self):
        assert isinstance(self.ba.reverse_transition_matrix, dict)
        assert len(self.ba.reverse_transition_matrix) > 0

    def test_reverse_predecessors(self):
        preds = self.ba.get_reverse_predecessors(2)
        assert 0 in preds
        assert 1 in preds

    def test_reverse_predecessors_unknown(self):
        assert self.ba.get_reverse_predecessors(99) == {}


class TestBackwardAnalyzerMultipleAbsorbing:
    """Test with multiple absorbing states to verify absorption probability split."""

    def setup_method(self):
        # 0 -> 1 (0.8) | 2 (0.2);  1 -> 3 (1.0);  absorbing: 2, 3
        tm = {0: {1: 0.8, 2: 0.2}, 1: {3: 1.0}}
        self.graph = EventGraph(tm)
        self.ba = BackwardAnalyzer(self.graph)

    def test_absorbing_states(self):
        assert set(self.ba.absorbing_states) == {2, 3}

    def test_absorption_from_0(self):
        probs = self.ba.get_all_absorption_probabilities(0)
        assert abs(probs[2] - 0.2) < 1e-6
        assert abs(probs[3] - 0.8) < 1e-6

    def test_absorption_from_1(self):
        probs = self.ba.get_all_absorption_probabilities(1)
        assert abs(probs[2] - 0.0) < 1e-6
        assert abs(probs[3] - 1.0) < 1e-6

    def test_all_sum_to_one(self):
        for t in self.ba.transient_states:
            probs = self.ba.get_all_absorption_probabilities(t)
            assert abs(sum(probs.values()) - 1.0) < 1e-6


class TestLSTMSimulator:
    """Test LSTMSimulator with a mock model factory."""

    def setup_method(self):
        from simulation.lstm_simulator import LSTMSimulator

        class MockModelFactory:
            """Returns a fixed distribution regardless of sequence."""
            def predict_proba(self, X):
                # 3 real events + END(3). Bias toward event 1, small chance of END.
                return {0: 0.1, 1: 0.6, 2: 0.1, 3: 0.2}

        self.sim = LSTMSimulator(
            model_factory=MockModelFactory(),
            unique_events_count=3,  # END = 3
            max_steps=20,
            seed=42,
        )

    def test_simulate_trajectory_starts_with_input(self):
        traj = self.sim.simulate_trajectory([0])
        assert traj[0] == 0

    def test_simulate_trajectory_bounded_by_max_steps(self):
        traj = self.sim.simulate_trajectory([0])
        assert len(traj) <= 21  # start + max 20 steps

    def test_simulate_batch_count(self):
        batch = self.sim.simulate_batch([0], 50)
        assert len(batch) == 50

    def test_simulate_from_prefix_preserves_prefix(self):
        results = self.sim.simulate_from_prefix([0, 1, 2], n_simulations=10)
        for traj in results:
            assert traj[:3] == [0, 1, 2]

    def test_simulate_from_prefix_empty_raises(self):
        with pytest.raises(ValueError):
            self.sim.simulate_from_prefix([], n_simulations=1)

    def test_trajectory_statistics(self):
        batch = self.sim.simulate_batch([0], 100)
        stats = self.sim.trajectory_statistics(batch)
        assert stats["count"] == 100
        assert stats["mean_length"] > 0
        total = sum(stats["end_event_distribution"].values())
        assert abs(total - 1.0) < 1e-6

    def test_next_event_distribution(self):
        dist = self.sim.next_event_distribution([0])
        assert isinstance(dist, dict)
        assert abs(sum(dist.values()) - 1.0) < 1e-6
