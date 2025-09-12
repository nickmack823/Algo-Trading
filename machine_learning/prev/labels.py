# -*- coding: utf-8 -*-
"""
======================== BIG PLAIN-ENGLISH OVERVIEW ========================
This module creates *classification labels* for ML trading, using the fixed-
horizon approach described in the paper (predict up / down / flat H bars ahead).

Key ML terms (simple definitions):
- Forward return: The % change from now (t) to some future bar (t+H).
- Horizon (H): How far into the future we look to compute the forward return.
- Threshold: A minimum absolute move required to call something "up" or "down".
  If the forward return is small (below |threshold|), we call it "flat" (0).
- Ternary labels: {-1, 0, +1} representing {short, hold/neutral, long}.
- Purging: We drop the last H rows from training because their future value
  is unknown at the time of labeling (thus their forward return is NaN).
- Leakage: Accidentally including future information in features/labels seen
  by the model at training time. We avoid it by:
  * building features from past-only values (shifted by 1 in the feature store),
  * and by dropping/purging rows whose forward returns are not yet known.

Design (high level):
1) We compute forward returns: fwd_ret[t] = (Close[t+H]/Close[t] - 1) - fees.
2) We convert each fwd_ret to a label with a symmetric threshold:
     +1 if  fwd_ret >  threshold
     -1 if  fwd_ret < -threshold
      0 otherwise
3) Optional: apply a volatility filter so labels are 0 during "dead" periods.
4) Align features (X) and labels (y) safely with no leakage.

This file is *strategy-agnostic* and reusable for ML baselines, NNFX-style
blends, candlestick strategies, etc.
==========================================================================="""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Internal: OHLCV column normalizer -----------------------------------
# Your indicator stack often uses capitalized OHLCV. We accept any case.
_OHLCV_ALIASES = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: _OHLCV_ALIASES.get(c.lower(), c) for c in df.columns}
    return df.rename(columns=cols)


# ---- Core API -------------------------------------------------------------


def add_forward_return(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    horizon: int = 5,
    fee_bps: float = 0.0,
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add a column with the fixed-horizon forward return.

    fwd_ret[t] = (price[t+horizon] / price[t] - 1) - fees
      where fees = fee_bps * 1e-4, i.e., 10 bps → 0.001 = 0.10%

    Notes:
    - The last `horizon` rows will be NaN (no future price). Those rows should be
      dropped before training/testing (see `make_classification_labels`).
    - This function MUTATES `df` by assigning the new column and returns the same df.

    Parameters
    ----------
    df : DataFrame with a DateTimeIndex and a price column.
    price_col : Which column to use (case-insensitive handled by normalizer).
    horizon : Number of bars ahead to measure the return.
    fee_bps : Simple constant cost per trade, in basis points (100 bps = 1%).
    out_col : Optional name for the forward return column. Defaults to "fwd_ret_h{H}".

    Returns
    -------
    df : The same DataFrame, with the forward return column added.
    """
    df = _normalize_ohlcv_columns(df)
    price_col = _OHLCV_ALIASES.get(
        price_col.lower(), price_col
    )  # <= make arg case-insensitive
    if price_col not in df.columns:
        raise KeyError(
            f"price_col '{price_col}' not found in DataFrame columns: {list(df.columns)}"
        )

    if out_col is None:
        out_col = f"fwd_ret_h{horizon}"

    fees = float(fee_bps) * 1e-4
    future_price = df[price_col].shift(-int(horizon))  # align future price at time t
    df[out_col] = (future_price / df[price_col] - 1.0) - fees
    return df


def make_classification_labels(
    df: pd.DataFrame,
    *,
    fwd_ret_col: str,
    threshold: float = 0.0,
    volatility_filter: Optional[Dict[str, float]] = None,
    out_col: Optional[str] = None,  # kept for API symmetry; label is returned as Series
) -> pd.Series:
    """
    Convert a forward return column into ternary labels {-1, 0, +1}.

    Rules:
      +1 if fwd_ret >  threshold
      -1 if fwd_ret < -threshold
       0 otherwise

    Optional volatility gating:
      volatility_filter = {"window": 30, "min_sigma": 1e-3}
      - Compute rolling std of simple close-to-close returns.
      - Where std < min_sigma, force label = 0 (avoid dead/low-vol periods).

    Purging:
      - Any row where `fwd_ret_col` is NaN (e.g., last H rows) becomes NaN in y.
        You should drop these rows before training (see align function).

    Returns
    -------
    y : pd.Series of dtype int with index aligned to df.
    """
    if fwd_ret_col not in df.columns:
        raise KeyError(f"fwd_ret_col '{fwd_ret_col}' not found in DataFrame.")

    # Base ternary labels
    y = pd.Series(0, index=df.index, dtype=int)
    y[df[fwd_ret_col] > float(threshold)] = 1
    y[df[fwd_ret_col] < -float(threshold)] = -1

    # Optional volatility filter
    if volatility_filter is not None:
        window = int(volatility_filter.get("window", 30))
        min_sigma = float(volatility_filter.get("min_sigma", 1e-3))
        df_norm = _normalize_ohlcv_columns(df)
        close_ret = df_norm["Close"].pct_change()
        sigma = close_ret.rolling(window=window, min_periods=max(2, window // 2)).std()
        y = y.where(sigma >= min_sigma, 0)

    # Purge unknown future area (end of series)
    y = y.where(df[fwd_ret_col].notna(), np.nan)

    return y


def class_weights_from_series(labels: pd.Series) -> Dict[int, float]:
    counts = labels.value_counts(dropna=True).to_dict()
    total = sum(counts.values()) if counts else 0
    weights: Dict[int, float] = {}
    for cls in (-1, 0, 1):
        c = counts.get(cls, 0)
        if c > 0:
            weights[cls] = total / (len(counts) * c)
    return weights


def align_features_and_labels(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align features (X) and labels (y) by index and drop NaN labels.

    Assumes your features were built *leakage-safe* (e.g., shifted by 1 bar in
    the feature store). This function simply:
      1) intersects indices,
      2) drops rows where y is NaN,
      3) returns aligned (Xc, yc).
    """
    common = X.index.intersection(y.index)
    Xc = X.loc[common]
    yc = y.loc[common]
    mask = yc.notna()
    return Xc[mask], yc[mask]


# ---- One-shot convenience -------------------------------------------------


def label_with_forward_returns(
    df_ohlcv: pd.DataFrame,
    *,
    horizon: int = 5,
    price_col: str = "Close",
    fee_bps: float = 0.0,
    threshold: float = 0.0,
    volatility_filter: Optional[Dict[str, float]] = None,
    fwd_ret_col: Optional[str] = None,
) -> pd.Series:
    """
    One-shot helper: compute forward returns and convert to {-1, 0, +1} labels.

    Returns
    -------
    y : pd.Series aligned to df_ohlcv index; NaN at the tail (purged area).
    """
    df = _normalize_ohlcv_columns(df_ohlcv.copy())
    if fwd_ret_col is None:
        fwd_ret_col = f"fwd_ret_h{horizon}"

    add_forward_return(
        df,
        price_col=price_col,
        horizon=horizon,
        fee_bps=fee_bps,
        out_col=fwd_ret_col,
    )

    y = make_classification_labels(
        df,
        fwd_ret_col=fwd_ret_col,
        threshold=threshold,
        volatility_filter=volatility_filter,
    )
    return y
