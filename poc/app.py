import sys
import os

# Ensure project root is on the path so that transformation/, models/, etc. can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.set_page_config(
    page_title="Predictive Process Mining",
    page_icon="🔮",
    layout="wide",
)

st.title("Predictive Process Mining — POC")
st.markdown(
    """
    ### Workflows

    Use the sidebar to navigate between pages:

    1. **Upload** — Upload a CSV event log, auto-detect columns, and transform into event sequences
    2. **Train** — Select a model type and forecasting task, train, and explore predictions
    3. **Simulate** — Visualize the process graph, run Monte Carlo or LSTM simulations

    ---

    **Quick start:** Upload `tests/synthetic_process_data.csv` (pre-transformed) on the Upload page
    using the *"Load Pre-transformed Dataset"* section, then proceed to Train.
    """
)
