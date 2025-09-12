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
- Caching: We save the computed features to Parquet so you don't recompute for
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
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import joblib
import numpy as np
import pandas as pd

from scripts.config import (
    FEATURES_CACHE_DIR,
    LABELS_CACHE_DIR,
    MODELS_CACHE_DIR,
    SCORES_CACHE_DIR,
)

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

        # Ensure the feature output aligns to the candle index strictly.
        if len(out) != len(df_ohlcv):
            raise ValueError(
                f"Feature '{spec.name}' returned length {len(out)} "
                f"but candles have {len(df_ohlcv)} rows. The feature function must return "
                f"one row per input bar."
            )

        # If the index doesn’t match, coerce it to the candle index (fail-soft but safe).
        if not out.index.equals(df_ohlcv.index):
            out = out.copy()
            out.index = df_ohlcv.index

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

    # Replace infs with NA; only drop rows if the caller asked for it
    feats = feats.replace([float("inf"), float("-inf")], pd.NA)
    if dropna:
        feats = feats.dropna()

    return feats


def load_or_build_features(
    symbol: str,
    timeframe: str,
    df_ohlcv: pd.DataFrame,
    feature_specs: Iterable[FeatureSpec],
    registry: Mapping[str, FeatureFunc],
    *,
    cache_dir: Union[str, Path] = FEATURES_CACHE_DIR,
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


def load_or_build_labels(
    symbol: str,
    timeframe: str,
    df_ohlcv: pd.DataFrame,
    *,
    horizon: int,
    fee_bps: float,
    threshold: float,
    volatility_filter: Optional[Dict[str, float]] = None,
    cache_dir: Union[str, Path] = LABELS_CACHE_DIR,
    extra_cache_params: Optional[Mapping[str, Union[str, int, float]]] = None,
    force_recompute: bool = False,
) -> Tuple[pd.Series, Path]:
    """
    Load ternary labels {-1,0,+1} from Parquet cache if available; otherwise build and cache.

    The key includes: (symbol, timeframe, candle signature, horizon, fee_bps, threshold, vol-filter, extra).
    """
    df = _normalize_ohlcv(df_ohlcv)
    candles_sig = _candles_signature(df)

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles_sig": candles_sig,
        "horizon": int(horizon),
        "fee_bps": float(fee_bps),
        "threshold": float(threshold),
        "vol_filter": dict(volatility_filter or {}),
        "extra": dict(extra_cache_params or {}),
    }
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[
        :16
    ]

    cache_dir = Path(cache_dir)
    parquet_path = cache_dir / symbol / timeframe / f"labels_{key}.parquet"

    if parquet_path.exists() and not force_recompute:
        y = _from_parquet(parquet_path)["label"].astype("float64")
        y.index = pd.to_datetime(y.index)
        return y, parquet_path

    # Build forward returns + labels (same primitives you already use)
    fwd_ret_col = f"fwd_ret_h{horizon}"
    add_forward_return(df, price_col="Close", horizon=horizon, fee_bps=fee_bps)
    y = make_classification_labels(
        df,
        fwd_ret_col=fwd_ret_col,
        threshold=threshold,
        volatility_filter=volatility_filter,
    )

    # Save as a single-column DataFrame for robust Parquet roundtrips
    _ensure_dir(parquet_path.parent)
    pd.DataFrame({"label": y}).to_parquet(parquet_path, engine="pyarrow", index=True)
    return y, parquet_path


def model_artifact_key(
    *,
    symbol: str,
    timeframe: str,
    features_parquet: Path,
    model_key: str,
    model_params: Mapping[str, Union[int, float, str]],
    label_horizon: int,
    label_threshold: float,
    fee_bps: float,
    train_start: str,
    train_end: str,
    extra: Optional[Mapping[str, Union[str, int, float]]] = None,
) -> str:
    """
    Deterministic key for saved model and scores.
    We include the *feature file hash* (already encodes candle signature and specs),
    plus model family/params and labeling config.
    """
    features_token = features_parquet.stem  # e.g., "features_1a2b3c4d..."
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "features_token": features_token,
        "model_key": (model_key or "").lower(),
        "model_params": dict(model_params or {}),
        "label_horizon": int(label_horizon),
        "label_threshold": float(label_threshold),
        "fee_bps": float(fee_bps),
        "train": {"start": str(train_start), "end": str(train_end)},
        "extra": dict(extra or {}),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def build_artifact_paths(symbol: str, timeframe: str, key: str) -> Tuple[Path, Path]:
    model_path = Path(MODELS_CACHE_DIR) / symbol / timeframe / f"{key}.joblib"
    scores_path = Path(SCORES_CACHE_DIR) / symbol / timeframe / f"scores_{key}.parquet"
    return model_path, scores_path


def load_model_if_exists(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def save_model(model, path: Path) -> None:
    _ensure_dir(path.parent)
    joblib.dump(model, path)


def load_scores_if_exists(path: Path) -> Optional[pd.Series]:
    if path.exists():
        df = pd.read_parquet(path, engine="pyarrow")
        s = df["score"]
        s.index = pd.to_datetime(s.index)
        return s
    return None


def save_scores(scores: pd.Series, path: Path) -> None:
    _ensure_dir(path.parent)
    pd.DataFrame({"score": scores}).to_parquet(path, engine="pyarrow", index=True)


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

ReturnMode = Literal["pct", "log"]


def ensure_forward_returns(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    horizons: Union[int, Sequence[int]] = 4,
    mode: ReturnMode = "pct",
    col_template: str = "fwd_ret_h{}",
    overwrite: bool = False,
    validate_index: bool = True,
) -> pd.DataFrame:
    """
    Ensure forward-return columns exist in `df` for the given horizons.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV-like frame indexed by time. Must include `price_col`.
    price_col : str, default "Close"
        Column to compute forward returns from.
    horizons : int | Sequence[int], default 4
        If int N -> creates columns for h=1..N. If sequence -> uses those exact horizons.
    mode : {"pct","log"}, default "pct"
        "pct": (P[t+h]/P[t]) - 1; "log": ln(P[t+h]) - ln(P[t]).
    col_template : str, default "fwd_ret_h{}"
        Name pattern; for h=1 -> "fwd_ret_h1".
    overwrite : bool, default False
        If False, existing columns are preserved.
    validate_index : bool, default True
        If True, checks monotonic increasing index.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with forward-return columns added.

    Raises
    ------
    KeyError
        If `price_col` is missing.
    ValueError
        If horizons invalid or index not monotonic (when validate_index=True).
    """
    if price_col not in df.columns:
        raise KeyError(
            f"ensure_forward_returns: price_col '{price_col}' not found in DataFrame columns."
        )

    if isinstance(horizons, int):
        if horizons <= 0:
            raise ValueError(
                "horizons must be a positive int or a non-empty sequence of positive ints."
            )
        hs = list(range(1, horizons + 1))
    else:
        hs = [int(h) for h in horizons]
        if not hs or any(h <= 0 for h in hs):
            raise ValueError("horizons sequence must contain positive integers.")

    mode = mode.lower()
    if mode not in ("pct", "log"):
        raise ValueError("mode must be 'pct' or 'log'.")

    if validate_index and not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index must be monotonic increasing for forward returns."
        )

    p = df[price_col].astype("float64")

    for h in hs:
        col = col_template.format(h)
        if (col in df.columns) and not overwrite:
            continue

        fwd = p.shift(-h)  # price at t+h
        if mode == "pct":
            ret = (fwd / p) - 1.0
        else:
            # log returns; NaNs propagate if non-positive prices exist
            ret = np.log(fwd) - np.log(p)

        df[col] = ret

    return df


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
    df = _normalize_ohlcv(df)
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
        df_norm = _normalize_ohlcv(df)
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
    df = _normalize_ohlcv(df_ohlcv.copy())
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


# -*- coding: utf-8 -*-
"""
Uniform ML model wrappers (LR / Linear SVM / Random Forest) for time-aware FX strategies.

Big picture
-----------
This module gives you a consistent interface for classic ML baselines on your
ternary labels {-1, 0, +1}, while staying leak-safe with your existing pipeline:

Upstream (already in your repo):
- Features are built from *past-only* indicator values and shifted by 1 bar
  to prevent look-ahead; results are cached to Parquet for reproducibility.  (see feature_store)
- Labels are ternary {-1,0,+1} using fixed-horizon forward returns with optional
  volatility gating; tail is purged; features & labels are aligned safely.      (see labels)
- Splits are strictly time-aware (walk-forward, expanding, blocked K-fold) and
  can use gap/embargo to mirror feature shift and minimize leakage.             (see splits)

This file only wraps sklearn models so Optuna/backtests don't care which model
you pick. Each wrapper implements:
    fit(X, y, sample_weight=None)
    predict_scores(X)         → continuous score (proba- or margin-based)
    predict_labels(X, threshold=0.0) → {-1, 0, +1} via dead-zone around 0.

Scoring rule
------------
For *probability-capable* models, the score is:
    score = P(y=+1) - P(y=-1)

For *margin* models (e.g., LinearSVC without calibration), the score is
the difference between decision margins for +1 and -1 (or a single signed margin
in binary cases).

You can turn on calibration (sigmoid) for well-behaved P(·) with `calibrate=True`.
Calibration uses the *last* `calibrate_frac` chunk of the training fold as a
chronological holdout, avoiding data leakage.

Notes
-----
- Works with missing classes in a given fold (e.g., no -1s): we degrade gracefully.
- Class weights default to an "auto" scheme derived from your label frequencies,
  or set class_weight='balanced'/dict/None. You can still pass sample_weight; sklearn
  multiplies them internally.
- For stability, LR uses l2 penalty and 'lbfgs' multinomial by default; SVM is LinearSVC;
  RF defaults are conservative.

Dependencies
------------
scikit-learn ≥ 1.2
"""
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ----------------------------- Utilities ----------------------------------
def _safe_check_X(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Ensure 2D numpy array and no NaNs/Infs."""
    if isinstance(X, pd.DataFrame):
        arr = X.values
    else:
        arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(
            "X contains NaN or Inf. Clean upstream (feature shifting/dropping)."
        )
    return arr


def _safe_check_y(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Ensure 1D numpy array and integer class labels (-1,0,1)."""
    if isinstance(y, pd.Series):
        arr = y.values
    else:
        arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {arr.shape}")
    return arr


def _combine_sample_and_class_weights(
    y: np.ndarray,
    *,
    class_weight: Union[str, Dict[int, float], None] = "auto",
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Optional[Dict[int, float]], Optional[np.ndarray]]:
    """
    Decide how to pass weights to sklearn. We return two things:
      1) class_weight parameter for the estimator (None / 'balanced' / dict)
      2) sample_weight vector to pass to fit (possibly multiplied by per-class weights)

    Strategy:
    - If class_weight == 'balanced' or dict: pass through as estimator param; keep sample_weight.
    - If class_weight == 'auto' (default): compute from y using your helper and pass as dict.
    - If class_weight is None: just pass sample_weight (or None).
    """
    out_class_weight = None
    out_sample_weight = None

    if class_weight == "balanced":
        out_class_weight = "balanced"
        out_sample_weight = sample_weight
    elif isinstance(class_weight, dict):
        out_class_weight = {int(k): float(v) for k, v in class_weight.items()}
        out_sample_weight = sample_weight
    elif class_weight == "auto":
        cw = class_weights_from_series(pd.Series(y))
        out_class_weight = cw if cw else None
        out_sample_weight = sample_weight
    elif class_weight is None:
        out_class_weight = None
        out_sample_weight = sample_weight
    else:
        raise ValueError(
            "class_weight must be one of: None, 'balanced', 'auto', or dict."
        )

    return out_class_weight, out_sample_weight


def _prob_diff(proba: np.ndarray, classes_: np.ndarray) -> np.ndarray:
    """
    Compute score = P(+1) - P(-1), safely handling missing classes in a fold.
    If P(+1) only: score = P(+1)
    If P(-1) only: score = -P(-1)
    If neither exists (degenerate): return zeros.
    """
    # classes_ like array([-1, 0, 1]) but may be subset
    idx_pos = np.where(classes_ == 1)[0]
    idx_neg = np.where(classes_ == -1)[0]

    p_pos = proba[:, idx_pos[0]] if idx_pos.size else 0.0
    p_neg = proba[:, idx_neg[0]] if idx_neg.size else 0.0

    if isinstance(p_pos, float) and isinstance(p_neg, float):
        return np.zeros(proba.shape[0], dtype=float)  # no +/- classes present
    if isinstance(p_pos, float):  # only -1 present
        return -p_neg
    if isinstance(p_neg, float):  # only +1 present
        return p_pos
    return p_pos - p_neg


def _margin_diff(margins: np.ndarray, classes_: np.ndarray) -> np.ndarray:
    """
    For decision_function outputs:
      - Binary case: margins shape = (n_samples,), sign indicates positive class.
      - Multi-class (OvR): margins shape = (n_samples, n_classes), where each column is
        the margin for "this class vs rest". Return margin(+1) - margin(-1) if both exist.
    """
    if margins.ndim == 1:
        # Binary. We need to know which class is treated as "positive" column.
        # sklearn encodes classes_ sorted ascending; in binary case, the decision_function
        # is for class classes_[1] vs classes_[0]. If classes_ = [-1, 1], positive is +1.
        # If classes_ = [0, 1], positive is +1; if classes_ = [-1, 0], positive is 0.
        # We want score wrt +/- classes. If +1 not present, return margins with best-effort sign.
        if classes_.size == 2:
            pos_class = classes_[1]  # sklearn convention
            if pos_class == 1:
                return margins
            elif pos_class == -1:
                return -margins
            else:
                # Binary without +1/-1 pair (e.g., {-1,0} or {0,1}). Map best-effort:
                return margins if pos_class in (1, 0) else -margins
        else:
            return margins  # unexpected shape; pass through
    else:
        # Multi-class OvR; pick +1 and -1 columns if they exist
        idx_pos = np.where(classes_ == 1)[0]
        idx_neg = np.where(classes_ == -1)[0]
        if idx_pos.size and idx_neg.size:
            return margins[:, idx_pos[0]] - margins[:, idx_neg[0]]
        elif idx_pos.size:
            return margins[:, idx_pos[0]]
        elif idx_neg.size:
            return -margins[:, idx_neg[0]]
        else:
            # No +/- classes; e.g., only {0}. Return zeros to be conservative.
            return np.zeros(margins.shape[0], dtype=float)


def _labels_from_scores(scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Map continuous scores to {-1, 0, +1} using a dead-zone |score| <= threshold ⇒ 0.
    """
    labels = np.zeros(scores.shape[0], dtype=int)
    labels[scores > float(threshold)] = 1
    labels[scores < -float(threshold)] = -1
    return labels


# ----------------------------- Base API -----------------------------------


class MLModel:
    """Abstract base wrapper."""

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None) -> "MLModel":
        raise NotImplementedError

    def predict_scores(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_labels(self, X, threshold: float = 0.0) -> np.ndarray:
        scores = self.predict_scores(X)
        return _labels_from_scores(scores, threshold=threshold)


# ----------------------------- Concrete Models ----------------------------


@dataclass
class LRConfig:
    C: float = 1.0
    penalty: str = "l2"
    multi_class: str = "auto"  # 'auto' -> multinomial when applicable
    max_iter: int = 200
    n_jobs: int = 1
    random_state: int = 42
    class_weight: Union[str, Dict[int, float], None] = "auto"  # 'auto' uses your helper
    calibrate: bool = False
    calibrate_frac: float = 0.2  # use the LAST frac of the train fold for calibration
    # If calibrate=True, CalibratedClassifierCV(method='sigmoid', cv='prefit') is used.


class LogisticRegressionModel(MLModel):
    def __init__(self, cfg: LRConfig = LRConfig()):
        self.cfg = cfg
        self._pipe: Optional[Pipeline] = None
        self._clf: Optional[BaseEstimator] = (
            None  # final predictor (pipe or calibrator)
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self, X, y, sample_weight: Optional[np.ndarray] = None
    ) -> "LogisticRegressionModel":
        X = _safe_check_X(X)
        y = _safe_check_y(y)

        cw_param, sw = _combine_sample_and_class_weights(
            y, class_weight=self.cfg.class_weight, sample_weight=sample_weight
        )

        # LR generally benefits from scaling
        base = LogisticRegression(
            C=self.cfg.C,
            penalty=self.cfg.penalty,
            multi_class=self.cfg.multi_class,
            max_iter=self.cfg.max_iter,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            class_weight=cw_param,
        )
        pipe = Pipeline(
            [("scaler", StandardScaler(with_mean=True, with_std=True)), ("lr", base)]
        )

        # Fit on full training (or train portion if calibrating)
        if self.cfg.calibrate:
            # Chronological split: tail portion for calibration
            n = X.shape[0]
            hold = max(1, int(round(self.cfg.calibrate_frac * n)))
            tr_end = n - hold
            if tr_end <= 0:
                raise ValueError("Not enough samples for calibration split.")

            pipe.fit(
                X[:tr_end],
                y[:tr_end],
                lr__sample_weight=sw[:tr_end] if sw is not None else None,
            )
            # Wrap in a calibrator using the tail as calibration set
            calibrator = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
            calibrator.fit(X[tr_end:], y[tr_end:])
            self._clf = calibrator
        else:
            pipe.fit(X, y, lr__sample_weight=sw if sw is not None else None)
            self._clf = pipe

        # Save for downstream access
        self._pipe = pipe
        # classes_ available on inner estimator; for calibrator, exposed at top level
        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict_scores(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted.")
        X = _safe_check_X(X)

        # Prefer probabilities (calibrated or native)
        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _prob_diff(proba, classes_)
        # Fallback to decision_function (rare for LR)
        if hasattr(self._clf, "decision_function"):
            margins = self._clf.decision_function(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _margin_diff(np.asarray(margins), classes_)
        # Extreme fallback: signed label of prediction (not ideal)
        preds = self._clf.predict(X)
        return preds.astype(float)


@dataclass
class LinearSVMConfig:
    C: float = 1.0
    max_iter: int = 5000
    random_state: int = 42
    class_weight: Union[str, Dict[int, float], None] = "auto"
    calibrate: bool = False
    calibrate_frac: float = 0.2


class LinearSVMModel(MLModel):
    """
    Linear SVM (OvR for multi-class). Scaled inputs; scores from margin (+1 vs -1) unless calibrated.
    """

    def __init__(self, cfg: LinearSVMConfig = LinearSVMConfig()):
        self.cfg = cfg
        self._pipe: Optional[Pipeline] = None
        self._clf: Optional[BaseEstimator] = None  # could be calibrator
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None) -> "LinearSVMModel":
        X = _safe_check_X(X)
        y = _safe_check_y(y)

        cw_param, sw = _combine_sample_and_class_weights(
            y, class_weight=self.cfg.class_weight, sample_weight=sample_weight
        )

        base = LinearSVC(
            C=self.cfg.C,
            max_iter=self.cfg.max_iter,
            random_state=self.cfg.random_state,
            class_weight=cw_param,
        )
        pipe = Pipeline(
            [("scaler", StandardScaler(with_mean=True, with_std=True)), ("svm", base)]
        )

        if self.cfg.calibrate:
            # Chronological holdout for calibration
            n = X.shape[0]
            hold = max(1, int(round(self.cfg.calibrate_frac * n)))
            tr_end = n - hold
            if tr_end <= 0:
                raise ValueError("Not enough samples for calibration split.")

            pipe.fit(
                X[:tr_end],
                y[:tr_end],
                svm__sample_weight=sw[:tr_end] if sw is not None else None,
            )
            calibrator = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
            calibrator.fit(X[tr_end:], y[tr_end:])
            self._clf = calibrator
        else:
            pipe.fit(X, y, svm__sample_weight=sw if sw is not None else None)
            self._clf = pipe

        self._pipe = pipe
        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict_scores(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted.")
        X = _safe_check_X(X)

        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _prob_diff(proba, classes_)
        # Uncalibrated: use decision function margins
        if hasattr(self._clf, "decision_function"):
            margins = self._clf.decision_function(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _margin_diff(np.asarray(margins), classes_)
        preds = self._clf.predict(X)
        return preds.astype(float)


@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    max_features: Union[int, float, str, None] = "sqrt"
    n_jobs: int = -1
    random_state: int = 42
    class_weight: Union[str, Dict[int, float], None] = "auto"
    calibrate: bool = False
    calibrate_frac: float = 0.2


class RandomForestModel(MLModel):
    def __init__(self, cfg: RFConfig = RFConfig()):
        self.cfg = cfg
        self._clf: Optional[BaseEstimator] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self, X, y, sample_weight: Optional[np.ndarray] = None
    ) -> "RandomForestModel":
        X = _safe_check_X(X)
        y = _safe_check_y(y)

        cw_param, sw = _combine_sample_and_class_weights(
            y, class_weight=self.cfg.class_weight, sample_weight=sample_weight
        )

        rf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            max_features=self.cfg.max_features,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            class_weight=cw_param,
        )

        if self.cfg.calibrate:
            # Chronological holdout for calibration
            n = X.shape[0]
            hold = max(1, int(round(self.cfg.calibrate_frac * n)))
            tr_end = n - hold
            if tr_end <= 0:
                raise ValueError("Not enough samples for calibration split.")

            rf.fit(
                X[:tr_end],
                y[:tr_end],
                sample_weight=sw[:tr_end] if sw is not None else None,
            )
            calibrator = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
            calibrator.fit(X[tr_end:], y[tr_end:])
            self._clf = calibrator
        else:
            rf.fit(X, y, sample_weight=sw if sw is not None else None)
            self._clf = rf

        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict_scores(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted.")
        X = _safe_check_X(X)

        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _prob_diff(proba, classes_)
        # Fallback: mean vote margin via decision_path not exposed -> use predicted class {-1,0,1}
        preds = self._clf.predict(X)
        return preds.astype(float)


# ----------------------------- Factory & typing ---------------------------

ModelLike = Union[LogisticRegressionModel, LinearSVMModel, RandomForestModel]


def make_model(name: str, **kwargs) -> ModelLike:
    """
    Factory for concise config from strings.

    Examples:
        model = make_model("lr", C=0.5, calibrate=True)
        model = make_model("svm", C=2.0, calibrate=False)
        model = make_model("rf", n_estimators=500, calibrate=True, calibrate_frac=0.15)
    """
    name = name.strip().lower()
    if name in ("lr", "logreg", "logistic", "logistic_regression"):
        cfg = LRConfig(**kwargs)
        return LogisticRegressionModel(cfg)
    if name in ("svm", "linear_svm", "linsvm", "linearsvc"):
        cfg = LinearSVMConfig(**kwargs)
        return LinearSVMModel(cfg)
    if name in ("rf", "randforest", "random_forest", "randomforest"):
        cfg = RFConfig(**kwargs)
        return RandomForestModel(cfg)
    raise ValueError(f"Unknown model name: {name}")


# ml_baselines/splits.py
# -*- coding: utf-8 -*-
"""
======================== BIG PLAIN-ENGLISH OVERVIEW ========================
Time-aware splitters for ML trading that respect chronology and minimize leakage.

What's here:
1) walk_forward_splits: fixed-length rolling windows (train -> test -> slide).
2) expanding_window_splits: growing train windows with rolling fixed test.
3) blocked_kfold_time_series: blocked, time-ordered K-Fold using np.array_split.
4) single_window: build one explicit split by sizes or date.
5) Utilities: index → integer positions; monotonic index assertion.

Leakage controls:
- gap: bars skipped between train end and test start (e.g., to mirror feature shift=1).
- embargo: bars skipped AFTER each test block before the NEXT training window can end,
  i.e., we ensure the next train does not encroach into recently tested data.

Fits with:
- feature_store: features are already shifted by 1 bar to avoid look-ahead.
- labels: forward-return labels purge the last H rows; alignment happens later.
==========================================================================="""
# ----------------------------- Types --------------------------------------

IndexLike = Union[pd.Index, Sequence]
Split = Tuple[np.ndarray, np.ndarray]  # (train_pos, test_pos)
SplitWithIndex = Tuple[
    np.ndarray, np.ndarray, pd.Index, pd.Index
]  # + (train_idx, test_idx)


# ----------------------------- Utilities ----------------------------------


def _assert_monotonic_index(idx: pd.Index) -> None:
    """
    Ensure the index is sorted ascending. We do NOT sort automatically to avoid
    silently desynchronizing with the caller's X/y ordering.
    """
    if not idx.is_monotonic_increasing:
        raise ValueError(
            "Index must be sorted ascending to use time-aware splits. "
            "Sort your DataFrame/Series first (e.g., df.sort_index())."
        )


def _as_positions(index_like: IndexLike) -> Tuple[pd.Index, np.ndarray]:
    """
    Return the original index (labels/timestamps) and integer positions [0..n-1].
    """
    idx = index_like if isinstance(index_like, pd.Index) else pd.Index(index_like)
    _assert_monotonic_index(idx)
    positions = np.arange(len(idx), dtype=int)
    return idx, positions


def _slice_window_positions(start: int, end: int, n: int) -> np.ndarray:
    """
    Return positions [start, end) clipped to [0, n].
    """
    start = max(0, int(start))
    end = min(n, int(end))
    if end <= start:
        return np.empty(0, dtype=int)
    return np.arange(start, end, dtype=int)


@dataclass(frozen=True)
class SplitParams:
    gap: int = 0  # bars skipped between train end and test start
    embargo: int = 0  # bars skipped AFTER test end before the next TRAIN can end


# ----------------------------- Validators ----------------------------------


def _validate_positive_int(name: str, value: Optional[int]) -> None:
    if value is None or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_have_data(n: int, train: int, test: int, gap: int) -> None:
    if train + gap + test > n:
        raise ValueError(
            f"Not enough data for one split: train({train}) + gap({gap}) + test({test}) > n({n})"
        )


# ----------------------------- Core splitters ----------------------------------


def walk_forward_splits(
    index_like: IndexLike,
    *,
    train_size: int,
    test_size: int,
    step: Optional[int] = None,
    params: SplitParams = SplitParams(),
    return_with_index: bool = False,
) -> Generator[Union[Split, SplitWithIndex], None, None]:
    """
    Fixed-size rolling windows:
      [TRAIN train_size] --gap--> [TEST test_size] --slide by step--> repeat

    Embargo semantics:
      After yielding a (train, test), the NEXT train start is advanced such that
      the next train window does not end before (end_test + embargo).

    Yields:
      (train_pos, test_pos) or (train_pos, test_pos, train_idx, test_idx)
    """
    _validate_positive_int("train_size", train_size)
    _validate_positive_int("test_size", test_size)
    if step is not None:
        _validate_positive_int("step", step)
    if params.gap < 0 or params.embargo < 0:
        raise ValueError("gap and embargo must be >= 0")

    idx, pos = _as_positions(index_like)
    n = len(idx)
    if step is None:
        step = test_size

    _validate_have_data(n, train_size, test_size, params.gap)

    start_train = 0
    while True:
        end_train = start_train + train_size
        start_test = end_train + params.gap
        end_test = start_test + test_size
        if end_test > n:
            break

        train_pos = _slice_window_positions(start_train, end_train, n)
        test_pos = _slice_window_positions(start_test, end_test, n)
        if len(train_pos) == 0 or len(test_pos) == 0:
            break

        if return_with_index:
            yield train_pos, test_pos, idx[train_pos], idx[test_pos]
        else:
            yield train_pos, test_pos

        # Next window starts by sliding 'step'
        start_train = start_train + step
        # Enforce embargo: next train must not end before (end_test + embargo)
        if params.embargo > 0:
            min_next_train_start = end_test + params.embargo - train_size
            start_train = max(start_train, min_next_train_start)


def expanding_window_splits(
    index_like: IndexLike,
    *,
    initial_train_size: int,
    test_size: int,
    step: Optional[int] = None,
    params: SplitParams = SplitParams(),
    return_with_index: bool = False,
) -> Generator[Union[Split, SplitWithIndex], None, None]:
    """
    Expanding TRAIN with fixed TEST:
      First: TRAIN [0, initial_train_size), TEST [train_end + gap, + test_size)
      Next: increase train_end by 'step' (default = test_size) and repeat.

    Embargo semantics:
      After each test, the next train_end is at least (end_test + embargo).
    """
    _validate_positive_int("initial_train_size", initial_train_size)
    _validate_positive_int("test_size", test_size)
    if step is not None:
        _validate_positive_int("step", step)
    if params.gap < 0 or params.embargo < 0:
        raise ValueError("gap and embargo must be >= 0")

    idx, pos = _as_positions(index_like)
    n = len(idx)
    if step is None:
        step = test_size

    _validate_have_data(n, initial_train_size, test_size, params.gap)

    train_end = initial_train_size
    while True:
        start_test = train_end + params.gap
        end_test = start_test + test_size
        if end_test > n:
            break

        train_pos = _slice_window_positions(0, train_end, n)
        test_pos = _slice_window_positions(start_test, end_test, n)
        if len(train_pos) == 0 or len(test_pos) == 0:
            break

        if return_with_index:
            yield train_pos, test_pos, idx[train_pos], idx[test_pos]
        else:
            yield train_pos, test_pos

        # Expand train window
        train_end = train_end + step
        if params.embargo > 0:
            train_end = max(train_end, end_test + params.embargo)


def blocked_kfold_time_series(
    index_like: IndexLike,
    *,
    n_splits: int,
    test_size: Optional[int] = None,
    params: SplitParams = SplitParams(),
    return_with_index: bool = False,
) -> Generator[Union[Split, SplitWithIndex], None, None]:
    """
    Time-ordered blocked K-Fold using np.array_split to avoid dropping the tail.

    If test_size is provided, we slice contiguous test blocks of approximately
    that size; otherwise we use np.array_split to split the full range into
    n_splits contiguous test blocks (the last blocks may differ by 1 element).

    For each fold k:
      TRAIN = all positions strictly BEFORE start_test - gap
      TEST  = the k-th contiguous block
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if params.gap < 0:
        raise ValueError("gap must be >= 0")

    idx, pos = _as_positions(index_like)
    n = len(idx)

    if test_size is not None:
        _validate_positive_int("test_size", test_size)
        # Construct test blocks of size ~test_size; include the tail as a final block
        edges = list(range(0, n, test_size))
        if edges[-1] != n:
            edges.append(n)
        blocks = [
            np.arange(edges[i], edges[i + 1], dtype=int) for i in range(len(edges) - 1)
        ]
        # If more blocks than n_splits, cap to first n_splits; else if fewer, raise.
        if len(blocks) < n_splits:
            raise ValueError(
                "With the given test_size, fewer than n_splits blocks are produced."
            )
        blocks = blocks[:n_splits]
    else:
        # Evenly split full range into n_splits contiguous blocks (diff <= 1)
        blocks = [
            np.asarray(b, dtype=int)
            for b in np.array_split(np.arange(n, dtype=int), n_splits)
        ]

    for test_pos in blocks:
        if len(test_pos) == 0:
            continue
        start_test = int(test_pos[0])
        end_train = max(0, start_test - params.gap)
        train_pos = _slice_window_positions(0, end_train, n)
        if len(train_pos) == 0:
            continue

        if return_with_index:
            yield train_pos, test_pos, idx[train_pos], idx[test_pos]
        else:
            yield train_pos, test_pos


def single_window(
    index_like: IndexLike,
    *,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    train_end_at: Optional[pd.Timestamp] = None,
    params: SplitParams = SplitParams(),
    return_with_index: bool = False,
) -> Union[Split, SplitWithIndex]:
    """
    Build one train/test split.

    Two ways:
      (A) Sizes: train_size + test_size
      (B) Date/label: train_end_at in index + test_size

    gap is respected between train end and test start.
    """
    if params.gap < 0:
        raise ValueError("gap must be >= 0")

    idx, pos = _as_positions(index_like)
    n = len(idx)

    if train_end_at is not None:
        if train_end_at not in idx:
            raise KeyError(f"train_end_at {train_end_at!r} is not in index.")
        _validate_positive_int("test_size", test_size)
        train_end_pos = int(np.searchsorted(idx, train_end_at, side="right"))
        start_test = train_end_pos + params.gap
        end_test = start_test + int(test_size)
        train_pos = _slice_window_positions(0, train_end_pos, n)
        test_pos = _slice_window_positions(start_test, end_test, n)

    else:
        _validate_positive_int("train_size", train_size)
        _validate_positive_int("test_size", test_size)
        _validate_have_data(n, int(train_size), int(test_size), params.gap)
        train_pos = _slice_window_positions(0, int(train_size), n)
        start_test = len(train_pos) + params.gap
        test_pos = _slice_window_positions(start_test, start_test + int(test_size), n)

    if return_with_index:
        return train_pos, test_pos, idx[train_pos], idx[test_pos]
    return train_pos, test_pos


# -*- coding: utf-8 -*-
"""
to_trades.py
------------
Score/position → TradePlan helpers tailored to your Backtester + StrategyCore.

Why this file exists
--------------------
Your Backtester expects strategy.generate_trade_plan(current_index, current_position, balance, quote_to_usd_rate)
to return a *list of TradePlan objects* for that single bar. It also handles stop-loss / take-profit exits
via PositionManager.update(...) on each open trade **before** consuming any explicit EXIT plans.

This module gives you two layers:
1) scores_to_positions(scores, thresholds, rules)    → a stable {-1,0,+1} position series with hysteresis/min-hold/cooldown
2) MLSignalPlanner(...).plans_at_index(i, ...)      → emit EXIT and/or ENTRY TradePlan(s) for bar i that your Backtester will consume

Design choices that match your runtime loop
-------------------------------------------
- We emit an EXIT plan *first* when a flip is detected (e.g., +1 → -1); by default we **do not** also emit a same-bar ENTRY,
  because your Backtester updates self.position only *after* processing plans, so a same-bar entry would be ignored.
  (You can enable same_bar_flip_entry=True to attempt an immediate re-entry on flips, but the Backtester will still block it.)
- ENTRY plans include direction, units, entry_price, stop_loss, optional take_profit, and a PositionManager configured for
  breakeven/trailing logic. The Backtester uses these fields to open trades and then manages SL/TP over time.

Minimal usage inside an ML Strategy
-----------------------------------
# in your ML strategy.prepare_data(...):
scores = model_scores_series  # index-aligned with your OHLCV df
positions = scores_to_positions(scores, thresholds=Thresholds(0.2,-0.2,hysteresis=0.05), rules=ExecutionRules(min_hold=2, cooldown=1))

self.planner = MLSignalPlanner(
    forex_pair=self.FOREX_PAIR,
    df=self.data_with_indicators,  # must include 'Close' and (optionally) an ATR column if you select atr-based risk
    positions=positions,
    risk=RiskParams(risk_pct=0.01, atr_col="ATR_14", atr_multiplier=1.5, tp_multiple=None, breakeven_trigger_pips=None, trail_start_pips=None, trail_distance_pips=None),
    same_bar_flip_entry=False,    # safer default with current Backtester
    source="ML",
)

# in your ML strategy.generate_trade_plan(...):
return self.planner.plans_at_index(current_index, current_position, balance, quote_to_usd_rate)

"""
# Prefer absolute import when inside your repo; fall back to local for notebooks/tests.
try:
    from scripts.strategies import strategy_core as sc
except Exception:  # pragma: no cover
    import strategy_core as sc  # type: ignore


# ------------------------------ Thresholding ----------------------------------


@dataclass(frozen=True)
class Thresholds:
    long_enter: float = 0.2
    short_enter: float = -0.2
    # Optional explicit exits; if None, use enter +/- hysteresis
    long_exit: Optional[float] = None
    short_exit: Optional[float] = None
    hysteresis: float = 0.05  # applied if exit thresholds are None


@dataclass(frozen=True)
class ExecutionRules:
    min_hold: int = 1  # bars to hold after entry before eligible to exit/flip
    cooldown: int = 0  # bars to stay flat after an exit
    max_hold: Optional[int] = None  # optional time stop (bars)


def scores_to_positions(
    scores: pd.Series,
    *,
    thresholds: Thresholds = Thresholds(),
    rules: ExecutionRules = ExecutionRules(),
) -> pd.Series:
    """
    Convert a continuous score series (e.g., P(+1)-P(-1)) into a stable {-1,0,+1} position series.
    - Hysteresis exits prevent whipsaw: long_exit defaults to long_enter-hysteresis; short_exit to short_enter+hysteresis
    - min_hold, cooldown, max_hold control churn and timing behavior.

    Returns
    -------
    pd.Series in {-1,0,+1}, same index as `scores`.
    """
    if not isinstance(scores, pd.Series):
        scores = pd.Series(scores)
    scores = scores.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    idx = scores.index
    assert idx.is_monotonic_increasing, "scores index must be sorted ascending"

    le = thresholds.long_enter
    se = thresholds.short_enter
    lx = (
        thresholds.long_exit
        if thresholds.long_exit is not None
        else le - abs(thresholds.hysteresis)
    )
    sx = (
        thresholds.short_exit
        if thresholds.short_exit is not None
        else se + abs(thresholds.hysteresis)
    )

    pos = np.zeros(len(scores), dtype=int)
    hold = 0
    cool = 0

    for i, s in enumerate(scores.values):
        if cool > 0:
            pos[i] = 0
            cool -= 1
            continue

        cur = pos[i - 1] if i > 0 else 0

        if cur == 0:
            if s >= le:
                cur = 1
                hold = 0
            elif s <= se:
                cur = -1
                hold = 0
            else:
                cur = 0
        else:
            hold += 1
            can_exit = hold >= max(1, rules.min_hold)

            if rules.max_hold is not None and hold >= rules.max_hold:
                cur = 0
                hold = 0
                cool = rules.cooldown
            elif cur == 1:
                if can_exit and s <= lx:
                    cur = 0
                    hold = 0
                    cool = rules.cooldown
                elif can_exit and s <= se:
                    # flip → exit now; entry handled by planner (often next bar)
                    cur = 0
                    hold = 0
                    cool = rules.cooldown
            else:  # cur == -1
                if can_exit and s >= sx:
                    cur = 0
                    hold = 0
                    cool = rules.cooldown
                elif can_exit and s >= le:
                    cur = 0
                    hold = 0
                    cool = rules.cooldown

        pos[i] = cur

    return pd.Series(pos, index=idx, name="position")


# ------------------------------ Risk / Sizing ---------------------------------


@dataclass(frozen=True)
class RiskParams:
    """
    Risk & exit configuration used for ENTRY TradePlans.
    - If `atr_col` is provided, we size stops & units from ATR × atr_multiplier.
    - Otherwise, we require `fixed_sl_pips` and use that for stops & sizing.
    - Optional TP and PM (breakeven/trailing) are in pips.
    """

    risk_pct: float = 0.01
    atr_col: Optional[str] = None
    atr_multiplier: float = 1.5  # when using ATR
    fixed_sl_pips: Optional[float] = None  # when not using ATR

    # Optional take-profit as multiple of stop distance (e.g., 1.5R)
    tp_multiple: Optional[float] = None

    # PositionManager config (all in pips relative to entry)
    breakeven_trigger_pips: Optional[float] = None
    trail_start_pips: Optional[float] = None
    trail_distance_pips: Optional[float] = None


def _pip_size(forex_pair: str) -> float:
    return sc.PositionCalculator.calculate_pip_size(forex_pair)


def _stop_distance_pips(
    risk: RiskParams, df: pd.DataFrame, i: int, forex_pair: str
) -> float:
    if risk.atr_col is not None:
        if risk.atr_col not in df.columns:
            raise KeyError(f"ATR column '{risk.atr_col}' not found in df.")
        atr_val = float(df[risk.atr_col].iloc[i])
        # Convert ATR value (in price units) to pips, then multiply
        atr_pips = sc.PositionCalculator.calculate_atr_pips(forex_pair, atr_val)
        return float(atr_pips * max(0.0, risk.atr_multiplier))
    if risk.fixed_sl_pips is None:
        raise ValueError("Provide either atr_col or fixed_sl_pips in RiskParams.")
    return float(risk.fixed_sl_pips)


def _units_for_entry(
    forex_pair: str,
    balance: float,
    risk: RiskParams,
    entry_price: float,
    df: pd.DataFrame,
    i: int,
    quote_to_usd_rate: Optional[float],
) -> int:
    # When ATR is used we can call the repo helper directly for units
    if risk.atr_col is not None:
        stop_pips = _stop_distance_pips(
            risk, df, i, forex_pair
        )  # respects risk.atr_multiplier
        dollars_per_pip_per_lot = sc.PositionCalculator.calculate_current_pip_value(
            forex_pair, 100_000, float(entry_price), quote_to_usd_rate
        )
        target_pip_value = sc.PositionCalculator.calculate_target_pip_value(
            int(balance), float(risk.risk_pct), float(stop_pips)
        )
        units = target_pip_value * 100_000 / dollars_per_pip_per_lot
        return int(round(units))

    # Otherwise, derive units from desired $ risk per trade and fixed pip stop
    dollars_per_pip_per_lot = sc.PositionCalculator.calculate_current_pip_value(
        forex_pair, 100_000, float(entry_price), quote_to_usd_rate
    )
    stop_pips = _stop_distance_pips(risk, df, i, forex_pair)
    target_pip_value = sc.PositionCalculator.calculate_target_pip_value(
        int(balance), float(risk.risk_pct), float(stop_pips)
    )
    units = target_pip_value * 100_000 / dollars_per_pip_per_lot
    return int(round(units))


def _stop_price_from_pips(
    forex_pair: str, direction: str, entry_price: float, stop_pips: float
) -> float:
    pipsz = _pip_size(forex_pair)
    if direction.upper() == "BUY":
        return float(entry_price - stop_pips * pipsz)
    else:
        return float(entry_price + stop_pips * pipsz)


def _tp_price_from_multiple(
    forex_pair: str,
    direction: str,
    entry_price: float,
    stop_pips: float,
    multiple: float,
) -> float:
    pipsz = _pip_size(forex_pair)
    if direction.upper() == "BUY":
        return float(entry_price + (stop_pips * multiple) * pipsz)
    else:
        return float(entry_price - (stop_pips * multiple) * pipsz)


# ------------------------------ Planner ---------------------------------------


class MLSignalPlanner:
    """
    Holds a precomputed position series and OHLCV(+ATR) DataFrame.
    Emits TradePlan(s) for a given bar index consistent with your Backtester.

    Typical flow inside a Strategy subclass:
        - compute scores → positions in prepare_data()
        - initialize planner with positions & risk config
        - in generate_trade_plan(i, ...), delegate to planner.plans_at_index(...)
    """

    def __init__(
        self,
        *,
        forex_pair: str,
        df: pd.DataFrame,
        positions: pd.Series,
        risk: RiskParams,
        source: str = "ML",
        same_bar_flip_entry: bool = False,
    ):
        assert "Close" in df.columns, "df must contain a 'Close' column"
        assert positions.index.equals(df.index), "positions index must equal df index"
        assert (
            positions.is_monotonic_increasing
        ), "positions index must be sorted ascending"
        self.forex_pair = forex_pair.replace("/", "")
        self.df = df
        self.pos = positions.astype(int).clip(-1, 1)
        self.risk = risk
        self.source = source
        self.same_bar_flip_entry = same_bar_flip_entry

    def _transition_at(self, i: int) -> int:
        """Return position transition at i: pos[i] - pos[i-1] (prev pos=0 for i==0)."""
        cur = int(self.pos.iloc[i])
        prev = int(self.pos.iloc[i - 1]) if i > 0 else 0
        return cur - prev

    def plans_at_index(
        self,
        i: int,
        current_position: int,
        balance: float,
        quote_to_usd_rate: Optional[float],
    ) -> List[sc.TradePlan]:
        """
        Emit a list of TradePlan(s) for bar i.
        Order: EXIT (if needed) then ENTRY (if allowed by same_bar_flip_entry and current_position).
        """
        plans: List[sc.TradePlan] = []
        transition = self._transition_at(i)

        # Flip or exit-to-flat ⇒ emit EXIT first (Backtester will close at current price w/ slippage)
        if (
            (transition < 0 and int(self.pos.iloc[i - 1] if i > 0 else 0) == 1)
            or (transition > 0 and int(self.pos.iloc[i - 1] if i > 0 else 0) == -1)
            or (transition == -1 and int(self.pos.iloc[i]) == 0)
            or (transition == 1 and int(self.pos.iloc[i]) == 0)
        ):
            plans.append(
                sc.TradePlan(
                    strategy=None,
                    direction="",  # ignored for EXIT
                    entry_price=float("nan"),
                    stop_loss=float("nan"),
                    tag="EXIT",
                    units=0,
                    source=self.source,
                )
            )

        # ENTRY on 0→+1 or 0→−1 (or on flip if same_bar_flip_entry=True)
        want_long = transition == 1 or (
            self.same_bar_flip_entry and transition < 0 and self.pos.iloc[i] == 1
        )
        want_short = transition == -1 or (
            self.same_bar_flip_entry and transition > 0 and self.pos.iloc[i] == -1
        )

        # Backtester will only open if currently flat; we still compute the plan.
        if want_long or want_short:
            direction = "BUY" if want_long else "SELL"
            close = float(self.df["Close"].iloc[i])
            stop_pips = float(
                _stop_distance_pips(self.risk, self.df, i, self.forex_pair)
            )
            stop_price = _stop_price_from_pips(
                self.forex_pair, direction, close, stop_pips
            )

            units = _units_for_entry(
                self.forex_pair,
                balance,
                self.risk,
                close,
                self.df,
                i,
                quote_to_usd_rate,
            )

            take_profit = None
            if self.risk.tp_multiple is not None:
                take_profit = _tp_price_from_multiple(
                    self.forex_pair,
                    direction,
                    close,
                    stop_pips,
                    float(self.risk.tp_multiple),
                )

            # Build PositionManager config in *price* units from pips settings
            pm_cfg = {}
            if self.risk.breakeven_trigger_pips is not None:
                pm_cfg["breakeven_trigger"] = float(
                    self.risk.breakeven_trigger_pips
                ) * _pip_size(self.forex_pair)
            if self.risk.trail_start_pips is not None:
                pm_cfg["trail_start"] = float(self.risk.trail_start_pips) * _pip_size(
                    self.forex_pair
                )
            if self.risk.trail_distance_pips is not None:
                pm_cfg["trail_distance"] = float(
                    self.risk.trail_distance_pips
                ) * _pip_size(self.forex_pair)

            pm = sc.PositionManager(
                entry_price=close,
                initial_stop_loss=stop_price,
                direction=direction,
                config=pm_cfg or None,
                take_profit=take_profit,
            )

            plans.append(
                sc.TradePlan(
                    strategy=None,  # Strategy subclass can set self.NAME before returning
                    direction=direction,  # 'BUY' or 'SELL'
                    entry_price=close,  # current close price as market entry
                    stop_loss=stop_price,
                    take_profit=take_profit,
                    units=units,
                    risk_pct=float(self.risk.risk_pct),
                    tag="",
                    position_manager=pm,
                    source=self.source,
                )
            )

        return plans
