import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd

from poc.components.column_detector import detect_columns
from transformation.transformation_type import TransformationType
from transformation.duration_strategy_type import DurationStrategyType
from transformation.merge_strategy_type import MergeStrategyType
from transformation.transformation_config import TransformationConfig
from transformation.transformation_manager import TransformationManager

st.title("Dataset Upload & Column Mapping")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV event log", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file, low_memory=False)
    st.session_state["raw_df"] = raw_df

if "raw_df" not in st.session_state:
    st.info("Upload a CSV to get started.")
    st.stop()

raw_df = st.session_state["raw_df"]
st.subheader("Preview (first 20 rows)")
st.dataframe(raw_df.head(20), use_container_width=True)

# --- Check if already transformed ---
existing_event_cols = [c for c in raw_df.columns if c.startswith("__EVENT_")]
is_pre_transformed = len(existing_event_cols) >= 2

if is_pre_transformed:
    st.info(
        f"This CSV already contains {len(existing_event_cols)} __EVENT_* columns — "
        "it appears to be pre-transformed. You can load it directly."
    )
    if st.button("Use as Pre-transformed Dataset", type="primary"):
        st.session_state["transformed_df"] = raw_df
        st.success(f"Loaded — {raw_df.shape[0]} cases, {raw_df.shape[1]} columns")

else:
    # --- Auto-detect columns ---
    detected = detect_columns(raw_df)
    all_columns = list(raw_df.columns)

    st.subheader("Column Mapping")
    col1, col2, col3 = st.columns(3)

    with col1:
        default_case = detected["case_id_columns"][0] if detected["case_id_columns"] else all_columns[0]
        case_id_col = st.selectbox(
            "Case ID column",
            all_columns,
            index=all_columns.index(default_case) if default_case in all_columns else 0,
        )

    with col2:
        default_event = detected["event_columns"][0] if detected["event_columns"] else all_columns[0]
        event_col = st.selectbox(
            "Event column",
            all_columns,
            index=all_columns.index(default_event) if default_event in all_columns else 0,
        )

    with col3:
        default_time = detected["time_columns"][0] if detected["time_columns"] else all_columns[0]
        time_col = st.selectbox(
            "Timestamp column",
            all_columns,
            index=all_columns.index(default_time) if default_time in all_columns else 0,
        )

    # --- Transformation parameters ---
    st.subheader("Transformation Parameters")
    p1, p2, p3 = st.columns(3)

    with p1:
        min_events = st.number_input("Min event sequence length", min_value=1, value=2)
    with p2:
        max_events = st.number_input("Max event sequence length", min_value=2, value=20)
    with p3:
        merge_strategy = st.selectbox(
            "Merge strategy",
            [ms.name for ms in MergeStrategyType],
            index=0,
        )

    # --- Transform ---
    if st.button("Transform Dataset", type="primary"):
        with st.spinner("Running transformation pipeline..."):
            config_dict = {
                "transformation_type": TransformationType.EVENTS.value,
                "case_id_column": case_id_col,
                "event_column": event_col,
                "min_event_sequence": min_events,
                "max_event_sequence": max_events,
                "duration_strategy": {
                    "type": DurationStrategyType.SINGLE_TIMESTAMP.value,
                    "time_column": time_col,
                },
                "merge_strategy": MergeStrategyType[merge_strategy].value,
                "event_prexix": "__EVENT_",
                "event_duration_prexix": "__DURATION_EVENT_",
            }
            try:
                config = TransformationConfig.from_dict(config_dict)
                manager = TransformationManager(config, raw_df.copy(), None)
                transformed_df = manager.execute()
                st.session_state["transformed_df"] = transformed_df
                st.success(f"Transformation complete — {transformed_df.shape[0]} cases, {transformed_df.shape[1]} columns")
            except Exception as e:
                st.error(f"Transformation failed: {e}")

# --- Display transformed summary ---
if "transformed_df" in st.session_state:
    df = st.session_state["transformed_df"]
    st.subheader("Transformed Dataset Summary")

    event_cols = [c for c in df.columns if c.startswith("__EVENT_")]
    duration_cols = [c for c in df.columns if c.startswith("__DURATION_EVENT_")]

    m1, m2, m3 = st.columns(3)
    # Count unique events across all event columns
    all_events = set()
    for col in event_cols:
        all_events.update(df[col].dropna().unique())
    all_events.discard("")

    m1.metric("Cases", df.shape[0])
    m2.metric("Unique Events", len(all_events))
    m3.metric("Max Sequence Length", len(event_cols))

    st.dataframe(df.head(20), use_container_width=True)
