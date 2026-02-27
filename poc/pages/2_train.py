import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from models.model_type import ModelType
from forecasting.forecasting_type import ForecastingType
from forecasting.config.forecasting_config import ForecastingConfig
from forecasting.base_forecasting import BaseForecasting
from models.config.model_config import ModelConfig
from prep.config.prep_config import PrepConfig

st.title("Model Training & Forecast Charts")

if "transformed_df" not in st.session_state:
    st.warning("No transformed dataset found. Go to the Upload page first.")
    st.stop()

df = st.session_state["transformed_df"]

# --- Configuration ---
st.subheader("Configuration")
c1, c2 = st.columns(2)

SUPPORTED_MODELS = {ModelType.MC: "Markov Chain", ModelType.LSTM: "LSTM"}

# MC only supports next_event; LSTM supports all three
_FORECASTING_BY_MODEL = {
    ModelType.MC: {ForecastingType.NEXT_EVENT: "Next Event"},
    ModelType.LSTM: {
        ForecastingType.NEXT_EVENT: "Next Event",
        ForecastingType.EVENT_DURATION: "Event Duration",
        ForecastingType.REMAINING_DURATION: "Remaining Duration",
    },
}

with c1:
    model_choice = st.selectbox(
        "Model type",
        list(SUPPORTED_MODELS.keys()),
        format_func=lambda x: SUPPORTED_MODELS[x],
    )

available_forecasting = _FORECASTING_BY_MODEL[model_choice]

with c2:
    forecasting_choice = st.selectbox(
        "Forecasting type",
        list(available_forecasting.keys()),
        format_func=lambda x: available_forecasting[x],
    )

# Pattern detection from dataset columns
event_cols = [c for c in df.columns if c.startswith("__EVENT_")]
duration_cols = [c for c in df.columns if c.startswith("__DURATION_EVENT_")]
event_pattern = "__EVENT_*" if event_cols else ""
duration_pattern = "__DURATION_EVENT_*" if duration_cols else ""

with st.expander("Advanced Parameters"):
    random_seq = st.checkbox("Use random sequence sampling", value=True)

# --- Train ---
if st.button("Train Model", type="primary"):
    if not event_cols:
        st.error("No __EVENT_* columns found in dataset.")
        st.stop()

    needs_duration = forecasting_choice in (
        ForecastingType.EVENT_DURATION,
        ForecastingType.REMAINING_DURATION,
    )
    if needs_duration and not duration_cols:
        st.error("Selected forecasting type requires __DURATION_EVENT_* columns.")
        st.stop()

    try:
        with st.spinner("Training model..."):
            forecasting_config = ForecastingConfig(
                forecasting_type=forecasting_choice,
                name_pattern=event_pattern,
                duration_pattern=duration_pattern,
                is_random_seq=random_seq,
            )
            prep_config = PrepConfig(
                include_all_data=False,
                drop_colum_names=[],
                drop_colum_patterns=[],
            )
            model_config = ModelConfig(
                model_type=model_choice,
                forecasting_config=forecasting_config,
                model_params={},
            )

            prediction_task = BaseForecasting.create_prediction_training(
                forecasting_config=forecasting_config,
                prep_config=prep_config,
                dataset=df,
            )
            model_factory = model_config.create_model()
            loss, accuracy = prediction_task.train(model_factory)

        # Store in session state
        st.session_state["model_factory"] = model_factory
        st.session_state["prediction_task"] = prediction_task
        st.session_state["train_loss"] = loss
        st.session_state["train_accuracy"] = accuracy
        st.session_state["model_type"] = model_choice
        st.session_state["forecasting_type"] = forecasting_choice

        st.success("Training complete!")
    except Exception as e:
        st.error(f"Training failed: {e}")

# --- Results ---
if "model_factory" not in st.session_state:
    st.info("Configure and train a model above.")
    st.stop()

model_factory = st.session_state["model_factory"]
prediction_task = st.session_state["prediction_task"]
loss = st.session_state["train_loss"]
accuracy = st.session_state["train_accuracy"]
model_type = st.session_state["model_type"]
forecasting_type = st.session_state["forecasting_type"]

# Metrics
st.subheader("Training Results")
r1, r2 = st.columns(2)
r1.metric("Loss", f"{loss:.4f}")
metric2_label = "Accuracy" if forecasting_type == ForecastingType.NEXT_EVENT else "Val Loss"
r2.metric(metric2_label, f"{accuracy:.4f}")

# --- Training History (LSTM only) ---
if model_type == ModelType.LSTM and hasattr(model_factory, "_ml_model") and model_factory._ml_model is not None:
    history = getattr(model_factory, "_training_history", None)
    if history is not None:
        st.subheader("Training History")
        hist_df = pd.DataFrame(history.history)
        fig_hist = px.line(hist_df, title="Training History (per epoch)")
        fig_hist.update_layout(xaxis_title="Epoch", yaxis_title="Value")
        st.plotly_chart(fig_hist, use_container_width=True)

# --- Prediction Explorer ---
st.subheader("Prediction Explorer")

if hasattr(prediction_task, "unique_events") and prediction_task.unique_events is not None:
    available_events = list(prediction_task.unique_events)
else:
    available_events = []
    for col in event_cols:
        available_events.extend(df[col].dropna().unique())
    available_events = sorted(set(e for e in available_events if e != ""))

prefix = st.multiselect("Select event sequence (prefix)", available_events, default=available_events[:1] if available_events else [])

if prefix and st.button("Predict"):
    if forecasting_type == ForecastingType.NEXT_EVENT:
        try:
            proba = prediction_task.predict_proba(model_factory=model_factory, X=prefix)
            predicted = prediction_task.predict(model_factory=model_factory, X=prefix)

            st.write(f"**Predicted next event:** {predicted}")

            proba_df = pd.DataFrame(
                {"Event": list(proba.keys()), "Probability": list(proba.values())}
            ).sort_values("Probability", ascending=False)

            fig = px.bar(
                proba_df,
                x="Event",
                y="Probability",
                title="Next Event Probability Distribution",
                color="Probability",
                color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    elif forecasting_type == ForecastingType.EVENT_DURATION:
        try:
            duration = prediction_task.predict(model_factory=model_factory, X=prefix)
            st.metric("Predicted Event Duration", f"{duration:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    elif forecasting_type == ForecastingType.REMAINING_DURATION:
        try:
            remaining = prediction_task.predict(model_factory=model_factory, X=prefix)
            st.metric("Predicted Remaining Duration", f"{remaining:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
