# strategies/ml_classification_strategy.py

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

# Feature store, labels, models, splits, and execution helpers
from machine_learning.core import load_model_if_exists  # NEW
from machine_learning.core import load_or_build_features  # NEW: use the feature cache
from machine_learning.core import load_or_build_labels  # NEW: use the label cache
from machine_learning.core import load_scores_if_exists  # NEW
from machine_learning.core import save_model  # NEW
from machine_learning.core import save_scores  # NEW
from machine_learning.core import (  # keep for non-cached paths; noqa; leakage-safe, cached  # noqa; uniform wrappers  # noqa; optional if you want to train on a window  # noqa; score→position→plans  # noqa
    ExecutionRules,
    FeatureSpec,
    LinearSVMConfig,
    LinearSVMModel,
    LogisticRegressionModel,
    LRConfig,
    MLSignalPlanner,
    RandomForestModel,
    RFConfig,
    RiskParams,
    SplitParams,
    Thresholds,
    add_forward_return,
    build_artifact_paths,
    build_features,
    ensure_forward_returns,
    model_artifact_key,
    scores_to_positions,
    single_window,
)

# Your core runtime / registry / trade objects
from scripts.config import (
    FEATURES_CACHE_DIR,
    LABELS_CACHE_DIR,
    MODELS_CACHE_DIR,
    SCORES_CACHE_DIR,
)
from scripts.strategies.strategy_core import (  # registry + BaseStrategy API  # noqa
    BaseStrategy,
    TradePlan,
    register_strategy,
)

ModelFactory = Callable[
    [], object
]  # returns a models.py wrapper with fit/predict_scores


def _align_X_y_index(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Safe alignment helper:
      - intersect on index,
      - drop NaNs/Infs in X,
      - drop any NaNs in y,
      - return aligned X, y, and the final index.
    """
    # Intersect indices and enforce monotonic order
    idx = X.index.intersection(y.index)
    idx = idx.sort_values()

    X = X.loc[idx]
    y = y.loc[idx]

    # Clean X strictly (models.py enforces finite matrix) and y (drop NaN labels)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    y = y.loc[X.index]
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    X = X.loc[y.index]  # final sync

    return X, y, X.index


@register_strategy(
    key="ml_classification",
    required_kwargs=[
        "forex_pair",
        "timeframe",
        "feature_specs",
        "registry",
    ],
    description="Generic ML classification strategy: build features → train model → scores→positions→TradePlans",
)
class MLClassificationStrategy(BaseStrategy):
    """
    A strategy-agnostic ML baseline:
      - features from arbitrary indicator registry (shifted to prevent leakage),
      - fixed-horizon forward-return labels (binary or ternary),
      - model wrappers (LR / LinearSVM / RF),
      - thresholds/hysteresis/min-hold/cooldown,
      - ATR- or fixed-pip risk with optional TP and PM (breakeven/trailing).
    """

    NAME = "MLClassificationStrategy"
    DESCRIPTION = "Generic ML classification strategy (indicator-agnostic)."

    # ---- constructor -------------------------------------------------------

    def __init__(
        self,
        *,
        forex_pair: str,
        timeframe: str,
        feature_specs: Iterable[FeatureSpec],
        indicator_registry: Mapping[
            str, Callable
        ],  # name -> feature function(df, params) -> Series/DataFrame
        # Labeling
        label_horizon: int = 1,
        label_threshold: float = 0.0,  # 0.0 yields binary up/down like the paper; >0 gives ternary with dead-zone
        fee_bps: float = 0.0,
        # Model selection: either pass a tiny factory OR choose via model_key + params
        model_factory: Optional[ModelFactory] = None,
        model_key: str = "lr",  # "lr" | "linearsvm" | "rf"
        model_params: Optional[Dict] = None,
        # Execution mapping scores → positions
        thresholds: Optional[Dict] = None,  # dict for Thresholds(...)
        rules: Optional[Dict] = None,  # dict for ExecutionRules(...)
        # Risk config for TradePlans
        risk: Optional[Dict] = None,  # dict for RiskParams(...)
        # Planner behavior
        same_bar_flip_entry: bool = False,  # keep False to match Backtester semantics
        split: Optional[Dict] = None,
    ):
        # 1) Build serializable identity kwargs for the DB config:
        param_kwargs = {
            # labeling / fees
            "label_horizon": int(label_horizon),
            "label_threshold": float(label_threshold),
            "fee_bps": float(fee_bps),
            # model
            "model_key": model_key,
            "model_params": dict(model_params or {}),
            # thresholds / rules / risk (as plain dicts, not objects)
            "thresholds": dict(thresholds or {}),
            "rules": dict(rules or {}),
            "risk": dict(risk or {}),
            # planner behavior
            "same_bar_flip_entry": bool(same_bar_flip_entry),
        }

        # ---- feature_specs identity (serialize to a JSON-safe, stable blob + short hash) ----
        def _serialize_feature_specs(specs):
            if not specs:
                return []
            out = []
            for s in specs:
                name = getattr(s, "name", None)
                prefix = getattr(s, "prefix", name)
                params = getattr(s, "params", {}) or {}
                params = {
                    k: (v.item() if hasattr(v, "item") else v)
                    for k, v in sorted(params.items())
                }
                out.append({"name": name, "prefix": prefix, "params": params})
            return out

        fs_blob = _serialize_feature_specs(feature_specs)
        fs_json = json.dumps(fs_blob, sort_keys=True, separators=(",", ":"))
        fs_hash = hashlib.sha1(fs_json.encode("utf-8")).hexdigest()[:12]

        param_kwargs["feature_specs"] = fs_blob
        param_kwargs["feature_hash"] = fs_hash

        # ---- let BaseStrategy turn **kwargs into PARAMETER_SETTINGS ----
        super().__init__(forex_pair=forex_pair, timeframe=timeframe, **param_kwargs)

        # Save user-config knobs
        self.feature_specs = list(feature_specs)
        self.indicator_registry = dict(indicator_registry)
        self.label_horizon = int(label_horizon)
        self.label_threshold = float(label_threshold)
        self.fee_bps = float(fee_bps)
        self.model_factory = model_factory
        self.model_key = model_key
        self.model_params = model_params or {}
        self.thresholds = Thresholds(**(thresholds or {}))
        self.rules = ExecutionRules(**(rules or {}))
        self.risk = RiskParams(**(risk or {}))
        self.same_bar_flip_entry = bool(same_bar_flip_entry)
        self.split = dict(split or {})

        # Runtime artifacts
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self._scores: Optional[pd.Series] = None
        self._positions: Optional[pd.Series] = None
        self._planner: Optional[MLSignalPlanner] = None

    # ---- BaseStrategy hooks ------------------------------------------------
    def prepare_data(
        self, historical_data: pd.DataFrame, use_cache: bool = True
    ) -> None:
        """
        End-to-end:
        1) Normalize index/timestamps for ML feature readiness.
        2) Build features (shifted by 1 bar to prevent look-ahead) WITHOUT nuking rows.
        3) Cut warmup using the latest first-valid across feature columns; gently impute stragglers.
        4) Build labels from fixed-horizon forward returns.
        5) Align X, y, split (train->fit; full->score).
        6) Convert scores→positions with thresholds/hysteresis/min-hold/cooldown.
        7) Initialize MLSignalPlanner to emit TradePlans in generate_trade_plan.
        """
        if historical_data is None or historical_data.empty:
            raise ValueError("historical_data is empty")

        # ---------------------------
        # (1) Normalize for ML feats
        # ---------------------------
        df = historical_data.copy()

        # Ensure a DatetimeIndex (rolling/warmups & helper code behave best with it)
        # If the index is not datetime but a 'Timestamp' column exists, use it.
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Timestamp" in df.columns:
                # Parse robustly; keep original order if already sorted
                ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)
                if ts.isna().all():
                    raise ValueError(
                        "Failed to parse 'Timestamp' to datetime for ML features."
                    )
                df = df.drop(columns=["Timestamp"]).set_index(ts)
            # else:
            # Fall back: treat current index as positional time; still works, but warn
            # print(
            #     "[prepare_data] Warning: non-datetime index and no 'Timestamp' column found."
            # )

        # Enforce monotonic increasing index and drop exact-duplicate timestamps (keep last)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]

        # Coerce core OHLCV columns to numeric where present (prevents object dtypes leaking in)
        for col in ("Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Quick sanity prints (keep during debugging; you can remove later)
        # print(df.columns.tolist())
        # print(df.dtypes)
        # print(df.head(3))
        # print(
        #     "df shape/index:", df.shape, type(df.index), df.index.min(), df.index.max()
        # )
        # print("has Close?", "Close" in df.columns)

        # ---------------------------------------------
        # (2) FEATURES: cache + shift=1, NO row dropna
        # ---------------------------------------------
        if use_cache:
            X, feats_path = load_or_build_features(
                symbol=self.FOREX_PAIR,
                timeframe=self.TIMEFRAME,
                df_ohlcv=df,
                feature_specs=self.feature_specs,
                registry=self.indicator_registry,
                shift_by=1,  # leakage safety
                dropna=False,  # <-- do NOT drop rows yet
            )
        else:
            X = build_features(
                df_ohlcv=df,
                feature_specs=self.feature_specs,
                indicator_registry=self.indicator_registry,
                shift_by=1,
                dropna=False,  # <-- do NOT drop rows yet
            )
            feats_path = (
                Path(FEATURES_CACHE_DIR)
                / self.FOREX_PAIR
                / self.TIMEFRAME
                / "features_uncached.parquet"
            )  # placeholder token only for artifact key stability

        # Normalize infinities → NaN (single pass)
        if not X.empty:
            X = X.replace([np.inf, -np.inf], np.nan)

        # Drop columns that are entirely NaN (unusable outputs)
        if not X.empty:
            all_nan_cols = [c for c in X.columns if X[c].isna().all()]
            if all_nan_cols:
                # print(
                #     f"[prepare_data] Dropping all-NaN feature columns: {all_nan_cols}"
                # )
                X = X.drop(columns=all_nan_cols)

        # Optional: drop ultra-sparse columns (e.g., >95% NaN across the index)
        if not X.empty:
            na_rate = X.isna().mean()
            too_sparse_cols = na_rate[na_rate > 0.95].index.tolist()
            if too_sparse_cols:
                # print(
                #     f"[prepare_data] Dropping ultra-sparse feature columns (>95% NaN): {too_sparse_cols}"
                # )
                X = X.drop(columns=too_sparse_cols)

        # Guard: if no usable columns remain
        if X.empty:
            col_list = list(na_rate.index) if "na_rate" in locals() else []
            print(
                "[prepare_data] Feature matrix empty after column pruning (no usable columns remain)."
            )
            print(f"[prepare_data] Columns considered: {col_list}")
            raise ValueError(
                "All feature columns were unusable (all-NaN or too sparse). "
                "Inspect indicator functions that produced these outputs."
            )

        # --------------------------------------------------------
        # (3) CUT WARMUP by latest first-valid across all columns
        # --------------------------------------------------------
        fvi = {c: X[c].first_valid_index() for c in X.columns}
        never_valid = [c for c, idx in fvi.items() if idx is None]
        if never_valid:
            # print(
            #     f"[prepare_data] Dropping columns with no valid values after pruning: {never_valid}"
            # )
            X = X.drop(columns=never_valid)
            fvi = {c: X[c].first_valid_index() for c in X.columns}

        if X.empty:
            print(
                "[prepare_data] Feature matrix empty after removing never-valid columns."
            )
            raise ValueError(
                "No feature columns have any valid values after warmup check."
            )

        # Compute cut position = latest first-valid among columns
        first_valid_positions = []
        for c, idx in fvi.items():
            if idx is not None:
                loc = X.index.get_loc(idx)
                # get_loc can return int or slice; normalize to integer start
                pos = loc.start if isinstance(loc, slice) else int(loc)
                first_valid_positions.append(pos)

        warmup_cut = max(first_valid_positions) if first_valid_positions else 0
        if warmup_cut > 0:
            # print(
            #     f"[prepare_data] Warmup cut: dropping first {warmup_cut} rows so all features are initialized."
            # )
            X = X.iloc[warmup_cut:]

        # Gentle impute any rare stragglers now that warmup is cut (keeps matrix dense for sklearn)
        if not X.empty:
            X = X.ffill().bfill()
            X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        # Final feature diagnostics
        if X.empty:
            print("[prepare_data] Feature matrix empty after warmup cut + cleaning.")
            raise ValueError(
                "All feature rows were dropped even after warmup cut. "
                "Check individual feature functions for persistent NaNs."
            )

        # print("X shape before align:", X.shape)
        # print(
        #     "X NaN rate (top 10):",
        #     X.isna().mean().sort_values(ascending=False).head(10),
        # )
        # print("Any ±inf in X?", np.isinf(X.to_numpy()).any())

        # --------------------------------
        # (4) LABELS: forward return build
        # --------------------------------
        h = int(getattr(self, "label_horizon", 1)) or 1
        ensure_forward_returns(
            df,
            price_col="Close",
            horizons=h,
            mode="pct",
            col_template="fwd_ret_h{}",
            overwrite=False,
            validate_index=True,
        )

        y, _labels_path = load_or_build_labels(
            symbol=self.FOREX_PAIR,
            timeframe=self.TIMEFRAME,
            df_ohlcv=df,
            horizon=self.label_horizon,
            fee_bps=self.fee_bps,
            threshold=self.label_threshold,
            volatility_filter=None,
            cache_dir=LABELS_CACHE_DIR,
        )

        # print("y shape before align:", y.shape, "NaN rate:", y.isna().mean())

        # -------------------------------------------
        # (5) ALIGN & derive train index window (ML)
        # -------------------------------------------
        X, y, aligned_index = _align_X_y_index(df, X, y)
        df_aligned = df.loc[aligned_index]

        # print("Aligned sizes:", X.shape, y.shape)
        # print(
        #     "Aligned index range:",
        #     X.index.min() if len(X) else None,
        #     "→",
        #     X.index.max() if len(X) else None,
        # )

        # --- temporal split (train→fit; full→score) ---
        train_end_idx = None
        if (self.split or {}).get("method", "temporal") == "temporal":
            if "train_end" in self.split:
                cut = pd.to_datetime(self.split["train_end"])
                # snap to existing index
                idx = aligned_index[aligned_index <= cut]
                if len(idx) > 0:
                    train_end_idx = idx.max()
            else:
                r = float(self.split.get("train_ratio", 0.8))
                r = min(max(r, 0.01), 0.99)
                k = int(np.floor(len(aligned_index) * r))
                if k >= 1:
                    train_end_idx = aligned_index[k - 1]

        if train_end_idx is None and len(aligned_index) > 0:
            train_end_idx = aligned_index[
                int(np.floor(len(aligned_index) * 0.8)) - 1
            ]  # fallback

        train_mask = (
            (aligned_index <= train_end_idx)
            if train_end_idx is not None
            else pd.Series(False, index=aligned_index)
        )
        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        # Artifact keys use train window for stability
        train_start = str(X_train.index.min()) if len(X_train) else ""
        train_end = str(X_train.index.max()) if len(X_train) else ""

        # print("Split config:", self.split)
        # print("train_end_idx:", train_end_idx)
        # print("X_train size:", X_train.shape, "y_train size:", y_train.shape)
        # print("train_start/end strings:", train_start, train_end)

        # 1) Structural invariants (hard fails if anything is off)
        assert len(X) and len(y), "Empty X or y after align"
        assert X.index.equals(y.index), "X/y indices must match exactly"
        assert (
            X.index.is_monotonic_increasing and not X.index.has_duplicates
        ), "Index must be sorted, unique"
        assert not np.isnan(X.to_numpy()).any(), "NaNs still present in X"
        assert not np.isinf(X.to_numpy()).any(), "Infs still present in X"
        assert not y.isna().any(), "NaNs still present in y"

        # Each feature has variance (not constant)
        var0 = [c for c in X.columns if X[c].var() == 0]
        assert not var0, f"Constant features detected: {var0}"

        # 2) Label sanity: horizon actually produces variability
        # print("y value counts:", y.value_counts(dropna=False).to_dict())

        # 3) Split sanity: no test rows in the train set
        if train_end_idx is not None:
            assert (
                X.loc[X.index <= train_end_idx].shape[0] == X_train.shape[0]
            ), "Train mask mismatch"
            assert (
                X.loc[X.index > train_end_idx].shape[0] + X_train.shape[0] == X.shape[0]
            ), "Train/Test partition mismatch"

        # 4) Leakage guard (spot check): features use t-1 while label uses fwd return from t
        # We can't re-derive all internals here, but at minimum confirm that your builder shifted by 1
        # print(
        #     "Leakage guard: features were built with shift_by=1 (see feature builder call)."
        # )

        # -------------------------
        # (6) Fit & score pipeline
        # -------------------------
        art_key = model_artifact_key(
            symbol=self.FOREX_PAIR,
            timeframe=self.TIMEFRAME,
            features_parquet=feats_path,
            model_key=self.model_key,
            model_params=self.model_params,
            label_horizon=self.label_horizon,
            label_threshold=self.label_threshold,
            fee_bps=self.fee_bps,
            train_start=train_start,
            train_end=train_end,
        )
        model_path, scores_path = build_artifact_paths(
            self.FOREX_PAIR, self.TIMEFRAME, art_key
        )

        scores = load_scores_if_exists(scores_path) if use_cache else None
        if scores is None:
            model = load_model_if_exists(model_path) if use_cache else None
            if model is None:
                model = self._make_model()
                # FIT ON TRAIN ONLY
                model.fit(X_train, y_train)
                if use_cache:
                    save_model(model, model_path)

            # Predict scores on the FULL aligned window (continuous series)
            scores = pd.Series(
                model.predict_scores(X), index=aligned_index, name="score"
            )
            if use_cache:
                save_scores(scores, scores_path)

        # ------------------------------------
        # (7) Scores → positions & final plan
        # ------------------------------------
        positions = scores_to_positions(
            scores, thresholds=self.thresholds, rules=self.rules
        )

        # Optionally mask pre-train positions to avoid trading before model training period
        if self.split.get("mask_pretrain_positions") and train_end_idx is not None:
            positions.loc[positions.index < train_end_idx] = 0

        self._planner = MLSignalPlanner(
            forex_pair=self.FOREX_PAIR,
            df=df_aligned,
            positions=positions,
            risk=self.risk,
            same_bar_flip_entry=self.same_bar_flip_entry,
            source="ML",
        )

        # Keep handles we’ll need in generate_trade_plan
        self._aligned_index = positions.index
        self._index_offset = int(df.index.get_indexer([self._aligned_index[0]])[0])
        self._n_aligned = int(len(self._aligned_index))

        # Full-length positions (zeros outside the aligned window)
        positions_full = pd.Series(0, index=df.index, dtype=int)
        positions_full.loc[self._aligned_index] = (
            positions.reindex(self._aligned_index).fillna(0).astype(int)
        )
        if self.split.get("mask_pretrain_positions") and train_end_idx is not None:
            positions_full.loc[positions_full.index < train_end_idx] = 0
        self._positions_full = positions_full

        # Respect pre-train masking inside the aligned window
        if self.split.get("mask_pretrain_positions") and train_end_idx is not None:
            positions_full.loc[positions_full.index < train_end_idx] = 0

        self._positions_full = positions_full

        assert len(X) and len(y), "Empty X or y after align"
        assert X.index.equals(y.index), "X/y indices must match exactly"
        assert (
            X.index.is_monotonic_increasing and not X.index.has_duplicates
        ), "Index must be sorted, unique"
        assert not np.isnan(X.to_numpy()).any(), "NaNs still present in X"
        assert not np.isinf(X.to_numpy()).any(), "Infs still present in X"
        assert not y.isna().any(), "NaNs still present in y"

        # Each feature has variance (not constant)
        var0 = [c for c in X.columns if X[c].var() == 0]
        assert not var0, f"Constant features detected: {var0}"

        # 2) Label sanity: horizon actually produces variability
        # print("y value counts:", y.value_counts(dropna=False).to_dict())

        # 3) Split sanity: no test rows in the train set
        if train_end_idx is not None:
            assert (
                X.loc[X.index <= train_end_idx].shape[0] == X_train.shape[0]
            ), "Train mask mismatch"
            assert (
                X.loc[X.index > train_end_idx].shape[0] + X_train.shape[0] == X.shape[0]
            ), "Train/Test partition mismatch"

        # 4) Leakage guard (spot check): features use t-1 while label uses fwd return from t
        # We can't re-derive all internals here, but at minimum confirm that your builder shifted by 1
        # print(
        #     "Leakage guard: features were built with shift_by=1 (see feature builder call)."
        # )

        self._X, self._y, self._scores, self._positions = X, y, scores, positions

    def get_cache_jobs(self) -> list[dict]:
        """Feature store handles Parquet caching internally; no indicator SQL cache jobs here."""
        return []

    def generate_trade_plan(
        self, current_index, current_position, balance, quote_to_usd_rate
    ):
        if self._planner is None:
            return []

        # Map raw backtester index (0..len(raw)-1) → local index (aligned window)
        idx_off = getattr(self, "_index_offset", 0)
        n_aligned = getattr(self, "_n_aligned", 0)
        local_idx = current_index - idx_off

        # Outside the aligned window → flat (no plans). This still lets the backtester
        # run the full range and compare fairly with NNFX.
        if local_idx < 0 or local_idx >= n_aligned:
            return []

        # Inside the aligned window → delegate to the planner
        plans = self._planner.plans_at_index(
            local_idx, current_position, balance, quote_to_usd_rate
        )
        for p in plans:
            p.strategy = self.NAME
        return plans

    # ---- internals ---------------------------------------------------------

    def _make_model(self):
        """Create a models.py wrapper instance either from a user factory or from (model_key, model_params)."""
        if callable(self.model_factory):
            return self.model_factory()

        key = (self.model_key or "lr").lower()
        mp = dict(self.model_params or {})

        if key == "lr":
            cfg = LRConfig(**mp)
            return LogisticRegressionModel(cfg)
        elif key in ("linearsvm", "svm"):
            cfg = LinearSVMConfig(**mp)
            return LinearSVMModel(cfg)
        elif key in ("rf", "randomforest", "random_forest"):
            cfg = RFConfig(**mp)
            return RandomForestModel(cfg)
        else:
            raise ValueError(f"Unknown model_key: {self.model_key!r}")
