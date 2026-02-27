import warnings

import pandas as pd


def _is_datetime_column(series: pd.Series, sample_size: int = 50) -> bool:
    """Check if a column can be parsed as datetime (excludes pure numerics)."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.head(sample_size)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pd.to_datetime(sample)
        return True
    except (ValueError, TypeError):
        return False


def _is_case_id_column(series: pd.Series) -> bool:
    """Heuristic: numeric or high-cardinality string that looks like an identifier."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    nunique = non_null.nunique()
    ratio = nunique / len(non_null)
    # Case IDs typically have high cardinality (many unique values per row)
    if ratio > 0.3:
        if pd.api.types.is_numeric_dtype(series):
            return True
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            return True
    return False


def _is_event_column(series: pd.Series, max_cardinality: int = 100) -> bool:
    """Heuristic: low-cardinality string column (categorical events)."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        return False
    nunique = non_null.nunique()
    return nunique <= max_cardinality


def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect column roles from a DataFrame.

    Returns dict with keys: 'time_columns', 'event_columns', 'case_id_columns'.
    Each value is a list of column names matching that role.
    """
    time_columns = []
    event_columns = []
    case_id_columns = []

    for col in df.columns:
        series = df[col]
        if _is_datetime_column(series):
            time_columns.append(col)
        elif _is_event_column(series):
            event_columns.append(col)
        elif _is_case_id_column(series):
            case_id_columns.append(col)

    return {
        "time_columns": time_columns,
        "event_columns": event_columns,
        "case_id_columns": case_id_columns,
    }
