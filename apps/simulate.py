# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import sys
import pandas as pd

from utils.file_util import load_json
from train.config.train_config import TrainConfig
from forecasting.base_forecasting import BaseForecasting
from simulation.event_graph import EventGraph
from simulation.forward_simulator import ForwardSimulator
from simulation.backward_analyzer import BackwardAnalyzer
from simulation.lstm_simulator import LSTMSimulator


def print_section(title):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")


def print_batch_stats(stats, idx_to_event, label):
    print(f"\n{label}:")
    print(f"  Cases: {stats['count']}")
    print(f"  Trajectory length: mean={stats['mean_length']:.2f}, "
          f"min={stats['min_length']}, max={stats['max_length']}, "
          f"std={stats['std_length']:.2f}")
    end_dist = {idx_to_event.get(e, e): round(p, 4) for e, p in stats['end_event_distribution'].items()}
    print(f"  End event distribution: {end_dist}")


def print_sample_trajectories(trajectories, idx_to_event, n=5, indent="  "):
    print(f"\n{indent}Sample trajectories ({min(n, len(trajectories))}):")
    for traj in trajectories[:n]:
        named = [idx_to_event.get(e, e) for e in traj]
        print(f"{indent}  {' -> '.join(named)}")


def run_prefix_simulations(sim, prefixes, n_prefix_sims, event_to_idx, idx_to_event):
    if not prefixes:
        return

    print_section(f"PREFIX SIMULATIONS ({n_prefix_sims} cases per prefix)")

    for prefix_events in prefixes:
        prefix_indices = [event_to_idx.get(e, -1) for e in prefix_events]
        if any(i == -1 for i in prefix_indices):
            print(f"\n  Prefix {prefix_events}: SKIPPED (unknown events)")
            continue

        results = sim.simulate_from_prefix(prefix_indices, n_simulations=n_prefix_sims)
        stats = sim.trajectory_statistics(results)
        end_dist = {idx_to_event.get(e, e): round(p, 4) for e, p in stats['end_event_distribution'].items()}

        print(f"\n  Prefix: {' -> '.join(prefix_events)}")
        print(f"    Trajectory length: mean={stats['mean_length']:.2f}")
        print(f"    End event distribution: {end_dist}")
        print_sample_trajectories(results, idx_to_event, n=3, indent="    ")


def run_markov_simulation(model_factory, prediction_task, sim_config):
    idx_to_event = prediction_task.idx_to_event
    event_to_idx = prediction_task.event_to_idx

    graph = EventGraph(model_factory.transition_matrix)

    print_section("EVENT GRAPH")
    print(f"Events: {[idx_to_event.get(e, e) for e in graph.events]}")
    print(f"Start events: {[idx_to_event.get(e, e) for e in graph.start_events]}")
    print(f"End events (absorbing): {[idx_to_event.get(e, e) for e in graph.end_events]}")
    print(f"Transient events: {[idx_to_event.get(e, e) for e in graph.transient_events]}")
    print(f"\nTransition probabilities:")
    for source, targets in graph.transition_matrix.items():
        source_name = idx_to_event.get(source, source)
        targets_named = {idx_to_event.get(t, t): round(p, 4) for t, p in targets.items()}
        print(f"  {source_name} -> {targets_named}")

    n_simulations = sim_config.get("n_simulations", 1000)
    max_steps = sim_config.get("max_steps", 100)
    seed = sim_config.get("seed", None)
    sim = ForwardSimulator(graph, max_steps=max_steps, seed=seed)

    print_section(f"MARKOV FORWARD SIMULATION ({n_simulations} cases)")
    for start_name in graph.start_events:
        start_display = idx_to_event.get(start_name, start_name)
        batch = sim.simulate_batch(start_name, n_simulations)
        stats = sim.trajectory_statistics(batch)
        print_batch_stats(stats, idx_to_event, f"From '{start_display}'")
        print_sample_trajectories(batch, idx_to_event)

    prefixes = sim_config.get("prefixes", [])
    n_prefix_sims = sim_config.get("n_prefix_simulations", 100)
    run_prefix_simulations(sim, prefixes, n_prefix_sims, event_to_idx, idx_to_event)

    # Backward analysis
    print_section("BACKWARD ANALYSIS — ABSORPTION PROBABILITIES")
    ba = BackwardAnalyzer(graph)
    absorbing_names = [idx_to_event.get(e, e) for e in ba.absorbing_states]
    print(f"Absorbing states: {absorbing_names}")
    print()
    for t in ba.transient_states:
        t_name = idx_to_event.get(t, t)
        probs = ba.get_all_absorption_probabilities(t)
        if probs:
            named_probs = {idx_to_event.get(e, e): round(p, 4) for e, p in probs.items()}
            print(f"  From '{t_name}': {named_probs}")

    print_section("REVERSE TRANSITION MATRIX — P(predecessor | event)")
    for event in graph.events:
        preds = ba.get_reverse_predecessors(event)
        if preds:
            event_name = idx_to_event.get(event, event)
            named_preds = {idx_to_event.get(p, p): round(v, 4) for p, v in preds.items()}
            print(f"  {event_name} <- {named_preds}")


def run_lstm_simulation(model_factory, prediction_task, sim_config):
    idx_to_event = prediction_task.idx_to_event
    event_to_idx = prediction_task.event_to_idx
    unique_count = len(prediction_task.unique_events)

    n_simulations = sim_config.get("n_simulations", 1000)
    max_steps = sim_config.get("max_steps", 100)
    seed = sim_config.get("seed", None)
    sim = LSTMSimulator(model_factory, unique_events_count=unique_count, max_steps=max_steps, seed=seed)

    # LSTM needs a starting sequence — use start events from the data
    start_events = sim_config.get("start_events", None)
    if start_events is None:
        # Default: pick events that appear first in training sequences
        start_events = [prediction_task.idx_to_event[0]]

    print_section(f"LSTM FORWARD SIMULATION ({n_simulations} cases)")
    for start_name in start_events:
        start_idx = event_to_idx.get(start_name, -1)
        if start_idx == -1:
            print(f"\n  Start event '{start_name}': SKIPPED (unknown)")
            continue

        batch = sim.simulate_batch([start_idx], n_simulations)
        stats = sim.trajectory_statistics(batch)
        print_batch_stats(stats, idx_to_event, f"From '{start_name}'")
        print_sample_trajectories(batch, idx_to_event)

    prefixes = sim_config.get("prefixes", [])
    n_prefix_sims = sim_config.get("n_prefix_simulations", 100)
    run_prefix_simulations(sim, prefixes, n_prefix_sims, event_to_idx, idx_to_event)


def main(config_json: str, dataset_csv: str):
    config_dict = load_json(config_json)
    sim_config = config_dict.get("simulation", {})
    train_config_dict = config_dict["train_config"]
    backend = sim_config.get("backend", "markov")

    config = TrainConfig.from_dict(train_config_dict)
    dataset = pd.read_csv(dataset_csv, low_memory=False)

    forecasting_config = config.model_config.forecasting_config
    prep_config = config.prep_config
    prediction_task = BaseForecasting.create_prediction_training(
        forecasting_config=forecasting_config,
        prep_config=prep_config,
        dataset=dataset
    )
    model_factory = config.model_config.create_model()
    loss, accuracy = prediction_task.train(model_factory)
    print(f"Model trained — Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    if backend == "lstm":
        run_lstm_simulation(model_factory, prediction_task, sim_config)
    else:
        if not hasattr(model_factory, 'transition_matrix') or model_factory.transition_matrix is None:
            print("Error: markov backend requires a model with a transition matrix (use model_type: MC)")
            sys.exit(1)
        run_markov_simulation(model_factory, prediction_task, sim_config)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m apps.simulate <Simulation Configuration JSON> <Dataset CSV>")
    else:
        main(config_json=sys.argv[1], dataset_csv=sys.argv[2])
