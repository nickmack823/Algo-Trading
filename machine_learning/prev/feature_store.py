# ml_baselines/feature_store.py
# -*- coding: utf-8 -*-
"""
======================== BIG PLAIN-ENGLISH OVERVIEW ========================
This module builds a *feature store* for ML trading strategies.

Key ML terms used here (simple definitions):
- Feature(s): Numeric columns your model learns from (e.g., RSI(14), ATR(14)/Close).
- Target/Label: The thing your model tries to predict (e.g., future up/down).
- Leakage: Accidentally giving the model info from the future. We *prevent* this
  by shifting indicator features by 1 bar so the model never "sees" the current bar's
  completed indicator value when making a decision for that same bar.
- Train/Test split: Separating data to evaluate generalization (handled elsewhere).
- Caching: We save the computed features to Parquet so you don’t recompute for
  every experiment (fast Optuna runs, consistent reproducibility).

Design:
- You pass in a candle DataFrame (index = timestamps, columns at least: 'open','high','low','close','volume').
- You pass a `feature_spec` describing which features to compute and their params.
- You pass a `registry` mapping feature names → callables so the store is pluggable.
- We compute, time-shift by 1 (to avoid leakage), clean NaNs, and save to Parquet.
- A small cache key ensures we reuse files when inputs/params match.

This file is strategy-agnostic. You can reuse it for NNFX, candlestick, or any ML strategy.
==========================================================================="""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import pandas as pd

from scripts.indicators.indicator_configs import IndicatorConfig

# REMOVE this line (it’s unused and non-generic):
# from scripts.indicators.calculation_functions import series_wma

# Add near the top, below imports:
_OHLCV_ALIASES = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Accept any casing; map to capitalized columns your indicators expect.
    cols = {c: _OHLCV_ALIASES.get(c.lower(), c) for c in df.columns}
    out = df.rename(columns=cols)
    return out


def _json_safe(obj):
    import numpy as _np

    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (_np.integer, _np.floating)):
        return obj.item()
    return obj


# ----------------------------- Types & Helpers -----------------------------

FeatureFunc = Callable[
    [pd.DataFrame, Mapping[str, Union[int, float, str]]], pd.Series | pd.DataFrame
]
"""
Each feature function takes:
  - df: OHLCV DataFrame
  - params: dict of parameters (e.g., {"period": 14})
and returns either:
  - a pd.Series (single feature), or
  - a pd.DataFrame (multiple related features, e.g., Bollinger bands)
Returned index MUST align to df.index.
"""


@dataclass(frozen=True)
class FeatureSpec:
    name: str  # key in the registry
    params: Mapping[str, Union[int, float, str]]  # parameters for the feature callable
    prefix: Optional[str] = None  # optional prefix for output columns (namespacing)

    def as_dict(self) -> dict:
        return {"name": self.name, "params": dict(self.params), "prefix": self.prefix}


def _hash_for_cache(
    symbol: str,
    timeframe: str,
    feature_specs: Iterable[FeatureSpec],
    candles_signature: str,
    extra_params: Optional[Mapping[str, Union[str, int, float]]] = None,
) -> str:
    """
    Build a stable hash to identify a unique feature set for caching.
    Uses symbol, timeframe, feature list, and a lightweight candle signature.
    """
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "features": [_json_safe(fs.as_dict()) for fs in feature_specs],
        "candles_sig": candles_signature,
        "extra": _json_safe(dict(extra_params or {})),
    }

    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _candles_signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    df = df.sort_index()
    first = str(df.index[0])
    last = str(df.index[-1])
    n = len(df)
    # Optional: enable a stronger digest if needed:
    # closes = df.get("Close", df.get("close"))
    # closes_digest = ""
    # if closes is not None:
    #     import numpy as np, hashlib
    #     closes_digest = hashlib.sha1(np.ascontiguousarray(closes.values).tobytes()).hexdigest()[:8]
    closes_digest = ""
    return f"{first}|{last}|{n}|{closes_digest}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Save features to Parquet (columnar, typed, compressed).
    """
    _ensure_dir(path.parent)
    df.to_parquet(path, engine="pyarrow", index=True)


def _from_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


# ----------------------------- Core API -----------------------------------


def build_features(
    df_ohlcv: pd.DataFrame,
    feature_specs: Iterable[FeatureSpec],
    indicator_registry: Mapping[str, FeatureFunc],
    *,
    shift_by: int = 1,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute feature columns according to feature_specs using the given registry.

    - We compute each feature (Series or DataFrame) and optionally prefix columns.
    - We then SHIFT the resulting features by `shift_by` bars (default 1) to avoid leakage.
      That means each row's features come strictly from the past.
    - Finally, we optionally drop rows with NaNs (typically the first 'max_lookback' bars).

    Parameters
    ----------
    df_ohlcv : DataFrame with at least ['open','high','low','close','volume'] and a DatetimeIndex.
    feature_specs : iterable of FeatureSpec defining which features to compute.
    registry : dict mapping feature names -> callable(df, params) -> Series|DataFrame.
    shift_by : int, how many bars to shift features forward in time to prevent leakage.
    dropna : bool, whether to drop rows containing NaNs after shift (recommended True).

    Returns
    -------
    DataFrame of features aligned to df_ohlcv.index.
    """
    # Normalize/sort always, even if called directly (not via loader)
    df_ohlcv = _normalize_ohlcv(df_ohlcv).sort_index()

    frames: List[pd.DataFrame] = []

    for spec in feature_specs:
        if spec.name not in indicator_registry:
            raise KeyError(f"Feature '{spec.name}' not found in registry.")
        func = indicator_registry[spec.name]
        out = func(df_ohlcv, spec.params)

        # Make DataFrame
        if isinstance(out, pd.Series):
            out = out.to_frame(spec.prefix or spec.name)
        elif isinstance(out, pd.DataFrame):
            # If no prefix provided, use spec.name to avoid name collisions.
            pf = spec.prefix or spec.name
            out = out.add_prefix(f"{pf}_")

        else:
            raise TypeError(
                f"Feature function '{spec.name}' must return Series or DataFrame."
            )

        frames.append(out)

    if not frames:
        raise ValueError("No features were produced. Check your feature_specs.")

    feats = pd.concat(frames, axis=1)

    # Shift to avoid lookahead/leakage: model for row t uses info strictly ≤ t-1.
    if shift_by:
        feats = feats.shift(shift_by)

    # Optionally drop NaNs introduced by lookbacks/shift.
    if dropna:
        feats = feats.dropna()

    # Ensure strictly numeric & finite (object columns can break)
    for col in feats.columns:
        if pd.api.types.is_object_dtype(feats[col].dtype):
            feats[col] = pd.to_numeric(feats[col], errors="coerce")
    feats = feats.replace([float("inf"), float("-inf")], pd.NA).dropna()

    return feats

    return feats


def load_or_build_features(
    symbol: str,
    timeframe: str,
    df_ohlcv: pd.DataFrame,
    feature_specs: Iterable[FeatureSpec],
    registry: Mapping[str, FeatureFunc],
    *,
    cache_dir: Union[str, Path] = "data/features",
    extra_cache_params: Optional[Mapping[str, Union[str, int, float]]] = None,
    force_recompute: bool = False,
    shift_by: int = 1,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, Path]:
    """
    Load features from Parquet cache if available; otherwise compute and cache.

    Returns (features_df, parquet_path)

    Caching key is derived from (symbol, timeframe, specs, candle signature, extra params).
    Use extra_cache_params to include anything else (e.g., your data vendor or resampling mode).
    """
    candles_sig = _candles_signature(df_ohlcv)
    key = _hash_for_cache(
        symbol, timeframe, feature_specs, candles_sig, extra_cache_params
    )
    cache_dir = Path(cache_dir)
    parquet_path = cache_dir / symbol / timeframe / f"features_{key}.parquet"

    if parquet_path.exists() and not force_recompute:
        feats = _from_parquet(parquet_path)
        return feats, parquet_path

    feats = build_features(
        df_ohlcv=_normalize_ohlcv(df_ohlcv),
        feature_specs=feature_specs,
        indicator_registry=registry,
        shift_by=shift_by,
        dropna=dropna,
    )
    _to_parquet(feats, parquet_path)
    return feats, parquet_path


# ----------------------------- Generic registry builders -----------------------------
# Replace the previous example adapters with the generic ones below.


def build_registry_from_indicator_configs(
    indicator_configs: Iterable[Mapping[str, object]],
) -> Mapping[str, FeatureFunc]:
    reg: Dict[str, FeatureFunc] = {}

    for cfg in indicator_configs:
        name = str(cfg["name"])
        func = cfg["function"]
        defaults = dict(
            cfg.get("parameters", {})
        )  # use config defaults if caller passes nothing

        def make_feature(fn: Callable, base: dict) -> FeatureFunc:
            def _feature(
                df: pd.DataFrame, params: Mapping[str, Union[int, float, str]] = None
            ):
                merged = {**base, **(params or {})}
                return fn(df, **merged)

            return _feature

        reg[name] = make_feature(func, defaults)

    return reg
