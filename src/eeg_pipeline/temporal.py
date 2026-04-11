from __future__ import annotations

import pandas as pd


def add_lagged_features(
    data: pd.DataFrame,
    feature_columns: list[str],
    group_col: str = "file_name",
    sort_col: str = "window_start_seconds",
    n_lags: int = 2,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    if n_lags <= 0:
        raise ValueError("n_lags must be positive")

    lagged_groups: list[pd.DataFrame] = []
    required_columns = [group_col, sort_col, *feature_columns]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns for lagged features: {missing}")

    for _, group in data.groupby(group_col, sort=False):
        ordered = group.sort_values(sort_col).copy()
        for lag in range(1, n_lags + 1):
            shifted = ordered[feature_columns].shift(lag)
            shifted.columns = [f"{column}__lag{lag}" for column in feature_columns]
            ordered = pd.concat([ordered, shifted], axis=1)

        if drop_incomplete:
            lag_columns = [f"{column}__lag{lag}" for lag in range(1, n_lags + 1) for column in feature_columns]
            ordered = ordered.dropna(subset=lag_columns)

        lagged_groups.append(ordered)

    return pd.concat(lagged_groups, ignore_index=True)
