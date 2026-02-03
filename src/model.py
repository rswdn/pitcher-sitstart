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


def add_reasons(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Add a human-readable explanation of top positive/negative contributions."""
    contributions = {}
    for feature, weight in HEURISTIC_WEIGHTS.items():
        if feature not in df.columns:
            continue
        feature_series = df[feature]
        filled = feature_series.fillna(feature_series.mean())
        if filled.isnull().all():
            filled = filled.fillna(0)
        z = _zscore(filled)
        contributions[feature] = weight * z

    def format_side(items):
        return ", ".join([f"{feat} ({val:+.2f})" for feat, val in items])

    reasons = []
    for idx in df.index:
        pos = []
        neg = []
        for feature, series in contributions.items():
            val = float(series.loc[idx])
            if val > 0:
                pos.append((feature, val))
            elif val < 0:
                neg.append((feature, val))
        pos = sorted(pos, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        neg = sorted(neg, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        parts = []
        if pos:
            parts.append(f"+ {format_side(pos)}")
        if neg:
            parts.append(f"- {format_side(neg)}")
        reasons.append(" | ".join(parts))
    df["reasons"] = reasons
    return df


def assign_tier(score: float) -> str:
    if score >= START_THRESHOLD:
        return "Start"
    if score >= STREAM_THRESHOLD:
        return "Stream"
    return "Sit"
