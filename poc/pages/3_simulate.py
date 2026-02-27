import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from models.model_type import ModelType
from simulation.event_graph import EventGraph
from simulation.forward_simulator import ForwardSimulator
from simulation.backward_analyzer import BackwardAnalyzer
from simulation.lstm_simulator import LSTMSimulator
from poc.components.graph_renderer import render_event_graph

st.title("Process Graph & Simulation")

if "model_factory" not in st.session_state or "prediction_task" not in st.session_state:
    st.warning("No trained model found. Go to the Train page first.")
    st.stop()

model_factory = st.session_state["model_factory"]
prediction_task = st.session_state["prediction_task"]
model_type = st.session_state["model_type"]

if prediction_task.unique_events is None:
    st.warning("Model attributes not initialized. Please re-train on the Train page.")
    st.stop()

idx_to_event = prediction_task.idx_to_event
event_to_idx = prediction_task.event_to_idx
unique_events = list(prediction_task.unique_events)

# ============================================================
# 3a. Event Graph Visualization (Markov Chain only)
# ============================================================
has_transition_matrix = hasattr(model_factory, "transition_matrix") and model_factory.transition_matrix is not None

if has_transition_matrix:
    st.subheader("Event Graph")
    graph = EventGraph(model_factory.transition_matrix)

    fig = render_event_graph(graph, idx_to_event)
    st.plotly_chart(fig, use_container_width=True)

    # Graph stats
    g1, g2, g3 = st.columns(3)
    g1.metric("Total Events", len(graph.events))
    start_names = [idx_to_event.get(e, str(e)) for e in graph.start_events]
    end_names = [idx_to_event.get(e, str(e)) for e in graph.end_events]
    g2.metric("Start Events", ", ".join(start_names))
    g3.metric("End Events", ", ".join(end_names))

    # Node inspector
    st.subheader("Node Inspector")
    selected_node_name = st.selectbox(
        "Select an event to inspect",
        unique_events,
    )
    if selected_node_name:
        selected_idx = event_to_idx[selected_node_name]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Outgoing Transitions**")
            successors = graph.get_successors(selected_idx)
            if successors:
                succ_df = pd.DataFrame(
                    {
                        "Target": [idx_to_event.get(t, str(t)) for t in successors],
                        "Probability": list(successors.values()),
                    }
                )
                st.dataframe(succ_df, use_container_width=True)
            else:
                st.write("Absorbing state (no outgoing transitions)")

        with col_b:
            st.markdown("**Absorption Probabilities**")
            try:
                ba = BackwardAnalyzer(graph)
                abs_probs = ba.get_all_absorption_probabilities(selected_idx)
                if abs_probs:
                    abs_df = pd.DataFrame(
                        {
                            "Absorbing State": [idx_to_event.get(e, str(e)) for e in abs_probs],
                            "Probability": list(abs_probs.values()),
                        }
                    )
                    st.dataframe(abs_df, use_container_width=True)
                else:
                    st.write("N/A (absorbing state)")
            except Exception as e:
                st.write(f"Could not compute: {e}")

    st.markdown("---")

# ============================================================
# 3b. Simulation Controls
# ============================================================
st.subheader("Simulation")

s1, s2, s3 = st.columns(3)
with s1:
    n_simulations = st.number_input("Number of simulations", min_value=1, max_value=10000, value=100)
with s2:
    max_steps = st.number_input("Max steps per trajectory", min_value=5, max_value=500, value=100)
with s3:
    if model_type == ModelType.LSTM:
        sim_backend = st.selectbox("Simulation backend", ["LSTM"], index=0)
    elif has_transition_matrix:
        sim_backend = st.selectbox("Simulation backend", ["Markov Chain"], index=0)
    else:
        sim_backend = "Markov Chain"

# Start event / prefix
sim_mode = st.radio("Simulation mode", ["From start event", "From prefix sequence"], horizontal=True)

if sim_mode == "From start event":
    start_event_name = st.selectbox("Start event", unique_events)
    prefix_events = [start_event_name] if start_event_name else []
else:
    prefix_events = st.multiselect("Prefix sequence", unique_events, default=unique_events[:1] if unique_events else [])

# --- Run Simulation ---
if st.button("Run Simulation", type="primary"):
    if not prefix_events:
        st.error("Select at least one event.")
        st.stop()

    prefix_indices = [event_to_idx[e] for e in prefix_events]

    with st.spinner(f"Simulating {n_simulations} trajectories..."):
        if sim_backend == "LSTM":
            sim = LSTMSimulator(
                model_factory,
                unique_events_count=len(unique_events),
                max_steps=max_steps,
            )
        else:
            if not has_transition_matrix:
                st.error("Markov backend requires a model with transition matrix (train with MC).")
                st.stop()
            graph = EventGraph(model_factory.transition_matrix)
            sim = ForwardSimulator(graph, max_steps=max_steps)

        if sim_mode == "From start event":
            if sim_backend == "LSTM":
                trajectories = sim.simulate_batch(prefix_indices, n_simulations)
            else:
                trajectories = sim.simulate_batch(prefix_indices[0], n_simulations)
        else:
            trajectories = sim.simulate_from_prefix(prefix_indices, n_simulations)

        stats = sim.trajectory_statistics(trajectories)

    # --- Display Results ---
    st.subheader("Simulation Results")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Simulated Cases", stats["count"])
    r2.metric("Mean Trajectory Length", f"{stats['mean_length']:.2f}")
    r3.metric("Min Length", stats["min_length"])
    r4.metric("Max Length", stats["max_length"])

    # End-event distribution chart
    end_dist = stats.get("end_event_distribution", {})
    if end_dist:
        st.subheader("End-Event Distribution")
        end_df = pd.DataFrame(
            {
                "End Event": [idx_to_event.get(e, str(e)) for e in end_dist],
                "Proportion": list(end_dist.values()),
            }
        ).sort_values("Proportion", ascending=False)

        fig_end = px.bar(
            end_df,
            x="End Event",
            y="Proportion",
            title="End-Event Distribution",
            color="Proportion",
            color_continuous_scale="Reds",
        )
        fig_end.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_end, use_container_width=True)

    # Trajectory length distribution
    st.subheader("Trajectory Length Distribution")
    lengths = [len(t) for t in trajectories]
    fig_len = px.histogram(
        x=lengths,
        nbins=min(30, max(lengths) - min(lengths) + 1) if lengths else 10,
        title="Trajectory Length Distribution",
        labels={"x": "Trajectory Length", "y": "Count"},
    )
    st.plotly_chart(fig_len, use_container_width=True)

    # Sample trajectories
    st.subheader("Sample Trajectories")
    n_samples = min(10, len(trajectories))
    for i, traj in enumerate(trajectories[:n_samples]):
        named = [idx_to_event.get(e, str(e)) for e in traj]
        st.code(f"#{i+1}: {' -> '.join(named)}", language=None)
