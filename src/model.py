"""Scoring logic for the sit/start analyzer."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from config import HEURISTIC_WEIGHTS, START_THRESHOLD, STREAM_THRESHOLD


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std == 0 or math.isnan(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mean) / std


def score_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stream score and tier for each pitcher row."""
    scores = pd.Series([0.0] * len(df), index=df.index)
    for feature, weight in HEURISTIC_WEIGHTS.items():
        if feature not in df.columns:
            continue
        feature_series = df[feature]
        filled = feature_series.fillna(feature_series.mean())
        if filled.isnull().all():
            filled = filled.fillna(0)
        z = _zscore(filled)
        scores += weight * z
    # squash to 0-1 for readability
    df["stream_score"] = 1 / (1 + np.exp(-scores))
    df["tier"] = df["stream_score"].apply(assign_tier)
    return df


def assign_tier(score: float) -> str:
    if score >= START_THRESHOLD:
        return "Start"
    if score >= STREAM_THRESHOLD:
        return "Stream"
    return "Sit"
