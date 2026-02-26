# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the Apache-2.0 license.
import sys
import json
import pandas as pd

from utils.file_util import load_json
from train.config.train_config import TrainConfig
from forecasting.base_forecasting import BaseForecasting
from simulation.event_graph import EventGraph
from simulation.forward_simulator import ForwardSimulator
from simulation.backward_analyzer import BackwardAnalyzer


def main(config_json: str, dataset_csv: str):
    config_dict = load_json(config_json)
    sim_config = config_dict.get("simulation", {})
    train_config_dict = config_dict["train_config"]

    config = TrainConfig.from_dict(train_config_dict)
    dataset = pd.read_csv(dataset_csv, low_memory=False)

    # Train Markov Chain to get transition matrix
    forecasting_config = config.model_config.forecasting_config
    prep_config = config.prep_config
    prediction_task = BaseForecasting.create_prediction_training(
        forecasting_config=forecasting_config,
        prep_config=prep_config,
        dataset=dataset
    )
    model_factory = config.model_config.create_model()
    loss, accuracy = prediction_task.train(model_factory)
    print(f"Markov Chain trained — Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    if not hasattr(model_factory, 'transition_matrix') or model_factory.transition_matrix is None:
        print("Error: simulation requires a model with a transition matrix (use MarkovChain)")
        sys.exit(1)

    # Build graph
    graph = EventGraph(model_factory.transition_matrix)
    idx_to_event = prediction_task.idx_to_event
    event_to_idx = prediction_task.event_to_idx

    print(f"\n{'='*60}")
    print("EVENT GRAPH")
    print(f"{'='*60}")
    print(f"Events: {[idx_to_event.get(e, e) for e in graph.events]}")
    print(f"Start events: {[idx_to_event.get(e, e) for e in graph.start_events]}")
    print(f"End events (absorbing): {[idx_to_event.get(e, e) for e in graph.end_events]}")
    print(f"Transient events: {[idx_to_event.get(e, e) for e in graph.transient_events]}")

    print(f"\nTransition probabilities:")
    for source, targets in graph.transition_matrix.items():
        source_name = idx_to_event.get(source, source)
        targets_named = {idx_to_event.get(t, t): round(p, 4) for t, p in targets.items()}
        print(f"  {source_name} -> {targets_named}")

    # Forward simulation
    n_simulations = sim_config.get("n_simulations", 1000)
    max_steps = sim_config.get("max_steps", 100)
    seed = sim_config.get("seed", None)

    sim = ForwardSimulator(graph, max_steps=max_steps, seed=seed)

    print(f"\n{'='*60}")
    print(f"FORWARD SIMULATION ({n_simulations} cases)")
    print(f"{'='*60}")

    for start_name in graph.start_events:
        start_display = idx_to_event.get(start_name, start_name)
        batch = sim.simulate_batch(start_name, n_simulations)
        stats = sim.trajectory_statistics(batch)

        print(f"\nFrom '{start_display}':")
        print(f"  Cases: {stats['count']}")
        print(f"  Trajectory length: mean={stats['mean_length']:.2f}, "
              f"min={stats['min_length']}, max={stats['max_length']}, "
              f"std={stats['std_length']:.2f}")
        end_dist = {idx_to_event.get(e, e): round(p, 4) for e, p in stats['end_event_distribution'].items()}
        print(f"  End event distribution: {end_dist}")

        print(f"\n  Sample trajectories (5):")
        for traj in batch[:5]:
            named = [idx_to_event.get(e, e) for e in traj]
            print(f"    {' -> '.join(named)}")

    # Prefix simulations
    prefixes = sim_config.get("prefixes", [])
    if prefixes:
        n_prefix_sims = sim_config.get("n_prefix_simulations", 100)
        print(f"\n{'='*60}")
        print(f"PREFIX SIMULATIONS ({n_prefix_sims} cases per prefix)")
        print(f"{'='*60}")

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

            print(f"    Sample continuations (3):")
            for traj in results[:3]:
                named = [idx_to_event.get(e, e) for e in traj]
                print(f"      {' -> '.join(named)}")

    # Backward analysis
    print(f"\n{'='*60}")
    print("BACKWARD ANALYSIS — ABSORPTION PROBABILITIES")
    print(f"{'='*60}")

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

    print(f"\n{'='*60}")
    print("REVERSE TRANSITION MATRIX — P(predecessor | event)")
    print(f"{'='*60}")
    for event in graph.events:
        preds = ba.get_reverse_predecessors(event)
        if preds:
            event_name = idx_to_event.get(event, event)
            named_preds = {idx_to_event.get(p, p): round(v, 4) for p, v in preds.items()}
            print(f"  {event_name} <- {named_preds}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m apps.simulate <Simulation Configuration JSON> <Dataset CSV>")
    else:
        main(config_json=sys.argv[1], dataset_csv=sys.argv[2])
