# scripts/trial_adapters/mabrouk_adapter.py
from __future__ import annotations

from scripts.indicators import indicator_configs

"""
MabroukAdapter
--------------
Adapter that wires your paper-specific ML strategy ("mabrouk_ml_2021") into the shared
Optuna flow. It mirrors NNFX's adapter shape so main.py can fan out studies the same way.

Why this matches your ML pipeline:
- FeatureSpec comes from the same API as feature_store.FeatureSpec (shift=1, cached) [see comments].
- Labels/horizon/threshold are managed by the strategy's internals, consistent with labels.py.
- Models are selected via 'model_key' and 'model_params' for your sklearn wrappers in models.py.
- Objective lifecycle is your shared run_objective_common: dedupe, backtest, pruning, DB queue.
- Positions/trade plans are driven by to_trades (Thresholds/Rules/Risk) with same_bar_flip_entry=False.

Phase overview:
- Phase-1: defaults (paper feature set), explore model + thresholds + risk.
- Phase-2: sample a feature set from a pool (NNFX-like exploration but for ML columns).
- Phase-3: rebuild top-N concrete strategies from DB and run fixed-window evaluations.
"""

import ast
import hashlib
import json
import multiprocessing as mp
from typing import Any, Dict, List

import optuna
import pandas as pd

# --- Optional import: use the same FeatureSpec class your feature_store expects.
# This aligns with ml_baselines/feature_store.FeatureSpec (name/params/prefix).
from machine_learning.core import FeatureSpec  # exact type your strategy uses
from scripts.config import TIMEFRAME_DATE_RANGES_PHASE_1_AND_2
from scripts.data.sql import BacktestSQLHelper
from scripts.strategies import strategy_core

from .base_adapter import (
    StrategyTrialAdapter,
    apply_execution_config_to_strategy,
    extract_execution_config_from_parameters,
    register_adapter,
    run_objective_common,
)

# ----------------------------- Feature pools -----------------------------------
# Paper list for fixed replication
_PAPER_FEATURE_NAMES = [
    "SMA",
    "EMA",
    "WMA",
    "RSI",
    "ADX",
    "CCI",
    "ROC",
    "BBANDS",
    "MACD",
]

_MABROUK_TRAIN_LOOKBACK_YEARS = 7

# Features known to be non-causal/repainting in their current implementation.
# Keep these out of ML optimization until they are rewritten as forward-only.
LOOKAHEAD_EXCLUDED_FEATURES = set()


def _build_train_then_test_context(
    timeframe: str,
    *,
    exploration_space: str = "parameterized",
    feature_space: dict | None = None,
    pool: str | None = None,
) -> dict:
    """
    Train on 7 years ending right before the selected test window, then evaluate on that window.

    - test window comes from TIMEFRAME_DATE_RANGES_PHASE_1_AND_2[timeframe]
    - loaded data spans [train_start, test_end]
    - split uses explicit train_end so pre-train positions can be masked reliably
    """
    tf_rng = TIMEFRAME_DATE_RANGES_PHASE_1_AND_2[timeframe]
    test_from, test_to = tf_rng["from_date"], tf_rng["to_date"]

    train_start = (
        pd.to_datetime(test_from)
        - pd.DateOffset(years=_MABROUK_TRAIN_LOOKBACK_YEARS)
    ).strftime("%Y-%m-%d")
    train_end = (pd.to_datetime(test_from) - pd.Timedelta(seconds=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    context = {
        "exploration_space": exploration_space,
        "feature_space": feature_space or {"min_features": 4, "max_features": 6},
        # Load full train+test range so model fits on history and trades only on test window.
        "date_window": {
            "absolute": {
                "from": train_start,
                "to": test_to,
            }
        },
        # Keep the explicit test window in context for debugging/reporting.
        "evaluation_window": {
            "from": test_from,
            "to": test_to,
        },
        "split": {
            "method": "temporal",
            "train_end": train_end,
            "mask_pretrain_positions": True,
        },
    }

    if pool is not None:
        context["pool"] = pool

    return context


def _build_pool_from_configs(configs, allowed_names=None):
    pool = {}
    for name, cfg in configs.items():  # note: configs is a dict now (name->cfg)
        if name in LOOKAHEAD_EXCLUDED_FEATURES:
            continue
        if allowed_names is not None and name not in allowed_names:
            continue
        parameter_space = dict(cfg.get("parameter_space", {}))
        if parameter_space:
            pool[name] = {"params": parameter_space, "category": cfg.get("category")}
        else:
            defaults = dict(cfg.get("parameters", {}))
            pool[name] = {
                "params": {k: [v] for k, v in defaults.items()},
                "category": cfg.get("category"),
            }
    return pool


# Build from the dict (indicator_configs already exports a by-name map)
ALL_FEATURE_POOL = _build_pool_from_configs(indicator_configs.all_indicators)
PAPER_FEATURE_POOL = _build_pool_from_configs(
    {
        k: v
        for k, v in indicator_configs.all_indicators.items()
        if k in _PAPER_FEATURE_NAMES
    }
)
MERGED_FEATURE_POOL = {**ALL_FEATURE_POOL, **PAPER_FEATURE_POOL}


# ----------------------------- Slot definitions -----------------------------
# Broad slot candidates (filtered later by the chosen POOL's keys)
def _names_by_cat(pool_dict, *cats):
    return [n for n, meta in pool_dict.items() if meta.get("category") in cats]


# Paper-faithful slot lists (unchanged)
_SLOT_TREND_PAPER = ["SMA", "EMA", "WMA", "ADX", "MACD"]
_SLOT_MOM_PAPER = ["RSI", "CCI", "ROC"]
_SLOT_VOL_PAPER = ["BBANDS"]


def _filter_available(slot_names, pool_dict):
    """Keep only names present in the active pool."""
    return [n for n in slot_names if n in pool_dict]


def _space_signature(values: list) -> str:
    """
    Stable short signature for a categorical choice set.
    Prevents Optuna dynamic-space collisions when choice lists evolve.
    """
    normalized = []
    for v in values:
        if hasattr(v, "item"):
            v = v.item()
        normalized.append(v)
    blob = json.dumps(
        normalized, ensure_ascii=True, sort_keys=False, separators=(",", ":")
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _suggest_feature(
    trial: optuna.Trial, pool_dict: dict, name: str, pool_tag: str
):
    meta = pool_dict[name]
    pspace = meta["params"]
    category = meta["category"]
    params = {}
    for pkey, pvals in pspace.items():
        pvals = list(pvals)
        sig = _space_signature(pvals)
        try:
            params[pkey] = trial.suggest_categorical(
                f"{pool_tag}.feat.{category}.{name}.{pkey}.{sig}",
                pvals,
            )
        except ValueError as e:
            print(f"Failed to sample {category} {name} {pkey} from {pvals}: {e}")
    return FeatureSpec(name=name, params=params, prefix=name)


class MabroukAdapter(StrategyTrialAdapter):
    """
    Paper-specific ML adapter for "Mabrouk2021".
    The underlying strategy inherits your generic MLClassificationStrategy and expects:
      - feature_specs: list[FeatureSpec] or None → strategy defaults if None
      - model_key/model_params: keys that map to models.py wrappers
      - thresholds/rules/risk: to_trades domain objects (strategy handles dicts)

    NOTE: Do NOT pass 'registry' here; your strategy factory injects it like NNFX.
    """

    key = "Mabrouk2021"

    # ------------------------ Objective (per trial) ----------------------------
    def objective(
        self,
        trial: optuna.Trial,
        forex_pair: str,
        timeframe: str,
        db_queue: mp.Queue,
        ack_dict: dict,
        context: dict,
    ) -> float:
        """
        Builds a concrete ML strategy for a single Optuna trial and executes the shared
        lifecycle (prepare_data → backtest → score → DB). We keep the search spaces
        compact to reduce overfit and to play nicely with your pruning rules.

        Context keys
        ------------
        exploration_space: "default" or "parameterized"
        feature_space   : bounds for Phase-2 feature count (min_features, max_features)
        """

        exploration = context.get("exploration_space", "default")

        # (A) Model family + params (must map to your models.py wrappers)
        # LR / LinearSVM / RF; params are stable defaults you already support.
        model_key = trial.suggest_categorical("model_key", ["lr", "linearsvm", "rf"])
        model_params: Dict[str, Any] = {}
        if model_key == "lr":
            model_params["C"] = trial.suggest_float("lr.C", 0.01, 10.0, log=True)
            model_params["max_iter"] = trial.suggest_int(
                "lr.max_iter", 200, 1000, step=100
            )
        elif model_key == "linearsvm":
            model_params["C"] = trial.suggest_float("svm.C", 0.01, 10.0, log=True)
            model_params["max_iter"] = trial.suggest_int(
                "svm.max_iter", 200, 2000, step=200
            )
        elif model_key == "rf":
            model_params["n_estimators"] = trial.suggest_int(
                "rf.n_estimators", 100, 600, step=50
            )
            model_params["max_depth"] = trial.suggest_int("rf.max_depth", 3, 12)

        mode = trial.suggest_categorical("th.mode", ["symmetric", "asymmetric"])

        if mode == "symmetric":
            base = trial.suggest_float("th.sym_base", 0.0, 0.40, step=0.05)
            long_enter = base
            short_enter = -base
        else:
            long_enter = trial.suggest_float("th.long_enter", 0.0, 0.40, step=0.05)
            short_enter = -trial.suggest_float(
                "th.short_enter_abs", 0.0, 0.40, step=0.05
            )

        hysteresis = trial.suggest_float("th.hysteresis", 0.0, 0.10, step=0.01)

        # Derive exits from enter ± hysteresis (keeps the space compact & consistent)
        long_exit = max(0.0, long_enter - hysteresis)
        short_exit = min(0.0, short_enter + hysteresis)

        thresholds = {
            "long_enter": long_enter,
            "short_enter": short_enter,
            "long_exit": long_exit,
            "short_exit": short_exit,
            "hysteresis": hysteresis,
        }

        # (C) Risk controls (ATR-based stops & sizing fit your MLSignalPlanner)
        # Decide whether TP is enabled
        tp_mode = trial.suggest_categorical("risk.tp_mode", ["none", "fixed"])

        if tp_mode == "none":
            tp_multiple = None
        else:
            tp_multiple = trial.suggest_float("risk.tp_multiple", 1.0, 3.0, step=0.5)

        risk = {
            "risk_pct": trial.suggest_float("risk.risk_pct", 0.005, 0.02, step=0.0025),
            "atr_col": "ATR_14",
            "atr_multiplier": trial.suggest_float("risk.atr_mult", 1.0, 3.0, step=0.25),
            "tp_multiple": tp_multiple,
        }

        # (D) Feature set (slot-constrained, pool-selectable):
        feature_specs = None
        if exploration == "parameterized":
            # Choose pool: "paper" for strict replication space, "broad" to explore full library
            pool_kind = context.get("pool") or trial.suggest_categorical(
                "pool.kind", ["paper", "broad"]
            )

            if pool_kind == "paper":
                POOL = PAPER_FEATURE_POOL
                slot_trend = _filter_available(_SLOT_TREND_PAPER, POOL)
                slot_mom = _filter_available(_SLOT_MOM_PAPER, POOL)
                slot_vol = _filter_available(_SLOT_VOL_PAPER, POOL)
                slot_volu = []
            else:
                # was: POOL = ALL_FEATURE_POOL
                POOL = MERGED_FEATURE_POOL  # ← use the merged set so paper names are guaranteed present
                # trend slot = baseline ∪ trend
                slot_trend = _names_by_cat(POOL, "baseline", "trend")
                slot_mom = _names_by_cat(POOL, "momentum")
                slot_vol = _names_by_cat(POOL, "volatility")
                slot_volu = _names_by_cat(POOL, "volume")  # optional wildcard source

            fs = context.get("feature_space", {"min_features": 4, "max_features": 6})
            min_k, max_k = fs["min_features"], fs["max_features"]
            target_k = trial.suggest_int("features.target_count", min_k, max_k)

            # sample slot counts
            n_trend = trial.suggest_int("slot.trend.count", 1, 2)
            n_mom = trial.suggest_int("slot.mom.count", 1, 2)
            n_vol = trial.suggest_int("slot.vol.count", 0, 1)
            n_wild = trial.suggest_int(
                "slot.wild.count", 0, 1
            )  # from trend/mom/vol/volu

            # adjust toward target_k via wildcard
            total = n_trend + n_mom + n_vol + n_wild
            if total < target_k:
                n_wild = min(n_wild + (target_k - total), 2)
            elif total > target_k and n_wild > 0:
                trim = min(total - target_k, n_wild)
                n_wild -= trim
            total = n_trend + n_mom + n_vol + n_wild

            chosen_names = []

            def _pick_from_stable(trial, slot_full, count, tag, pool_tag):
                """
                Pick `count` names from a STABLE choices list (slot_full) using Optuna.
                - The parameter name includes pool_tag and position index to keep distribution fixed.
                - We DO NOT filter choices before suggestion (avoids dynamic value space).
                - After sampling, we enforce uniqueness and pad deterministically from slot_full.
                """
                slot_full = list(slot_full)
                slot_sig = _space_signature(slot_full)
                # 1) raw picks with stable choices
                raw = []
                for i in range(count):
                    param_name = f"{pool_tag}.slot.{tag}.{i}.name.{slot_sig}"
                    choice = trial.suggest_categorical(param_name, slot_full)
                    raw.append(choice)

                # 2) enforce uniqueness (preserve order of first occurrence)
                uniq = []
                seen = set()
                for c in raw:
                    if c not in seen:
                        uniq.append(c)
                        seen.add(c)

                # 3) pad deterministically from slot_full if we lost items due to dupes
                if len(uniq) < count:
                    for c in slot_full:
                        if c not in seen:
                            uniq.append(c)
                            seen.add(c)
                            if len(uniq) == count:
                                break

                return uniq

            pool_tag = f"pool.{pool_kind}"

            trend_picked = _pick_from_stable(
                trial, slot_trend, n_trend, "trend", pool_tag
            )
            mom_picked = _pick_from_stable(trial, slot_mom, n_mom, "mom", pool_tag)
            vol_picked = _pick_from_stable(trial, slot_vol, n_vol, "vol", pool_tag)

            # Wildcard: sample from a STABLE union list (no dynamic filtering)
            any_slot_union = list(
                dict.fromkeys(slot_trend + slot_mom + slot_vol + slot_volu)
            )
            wild_picked = _pick_from_stable(
                trial, any_slot_union, n_wild, "wild", pool_tag
            )

            picked_all = trend_picked + mom_picked + vol_picked + wild_picked
            if len(picked_all) > target_k:
                # trim in stable order: wild -> vol -> mom -> trend
                order = wild_picked + vol_picked + mom_picked + trend_picked
                picked_all = order[:target_k]

            # Remove cross-slot duplicates (wildcard can reselect an existing feature).
            deduped = []
            seen = set()
            for fname in picked_all:
                if fname not in seen:
                    deduped.append(fname)
                    seen.add(fname)

            # Keep feature count stable after dedupe by padding in deterministic order.
            if len(deduped) < target_k:
                for fname in any_slot_union:
                    if fname not in seen:
                        deduped.append(fname)
                        seen.add(fname)
                        if len(deduped) == target_k:
                            break

            picked_all = deduped[:target_k]
            feature_specs = [
                _suggest_feature(trial, POOL, name, pool_tag) for name in picked_all
            ]

        # Strategy factory: mirror NNFX
        def build_strategy():
            return strategy_core.create_strategy_from_kwargs(
                MabroukAdapter.key,
                forex_pair=forex_pair,
                timeframe=timeframe,
                feature_specs=feature_specs,  # must be FeatureSpec objects, not dicts
                model_key=model_key,
                model_params=model_params,
                thresholds=thresholds,  # dicts are fine, MLClassificationStrategy wraps them
                # Rules left to strategy defaults; set here if you want to tune them.
                risk=risk,
                same_bar_flip_entry=False,  # safer default with current Backtester
                split=context.get("split"),
            )

        # Shared lifecycle (dedupe, prepare_data, indicator cache queue, backtest, score, prune)
        return run_objective_common(
            build_strategy, trial, forex_pair, timeframe, db_queue, ack_dict, context
        )

    # -------------------------- Phase builders ---------------------------------
    def build_phase1_args(
        self,
        pairs,
        timeframes,
        seeds,
        trials_by_timeframe,
        exploration_space="parameterized",  # switched from "default"
        phase_name="phase1",
    ):
        """
        Phase-1: parameterized — sample slots from the paper's nine indicators and tune
        model/threshold/risk across pairs/timeframes.
        """
        args = []
        for pair in pairs:
            for tf in timeframes:
                n = trials_by_timeframe[tf]
                for seed in seeds:
                    context = _build_train_then_test_context(
                        timeframe=tf,
                        exploration_space="parameterized",
                        feature_space={"min_features": 4, "max_features": 6},
                        pool="broad",
                    )
                    args.append(
                        (
                            pair,
                            tf,
                            n,
                            exploration_space,
                            phase_name,
                            seed,
                            self.key,
                            context,
                        )
                    )
        return args

    def build_phase2_args(
        self,
        pairs,
        timeframes,
        seeds,
        trials_by_timeframe,
        top_percent,
        exploration_space="parameterized",
        phase_name="phase2",
    ):
        """
        Phase-2: sample FeatureSpec[] from FEATURE_POOL and parametrize indicators/models.
        This mirrors the "iterate over library" spirit you have in NNFX, but for ML features.
        """
        args: List[tuple] = []
        for pair in pairs:
            for tf in timeframes:
                n = trials_by_timeframe[tf]
                for seed in seeds:
                    context = _build_train_then_test_context(
                        timeframe=tf,
                        exploration_space="parameterized",
                        feature_space={"min_features": 4, "max_features": 6},
                    )
                    args.append(
                        (
                            pair,
                            tf,
                            n,
                            exploration_space,
                            phase_name,
                            seed,
                            self.key,
                            context,
                        )
                    )
        return args

    def build_phase3_args(
        self,
        pairs: List[str],
        timeframes: List[str],
        top_n: int,
    ) -> List[tuple]:
        """
        Phase-3: rebuild top-N Mabrouk configs from DB and return fixed-eval jobs as
        (pair, timeframe, strategy_obj) tuples for run_fixed_strategy_evaluation.

        We parse Parameters defensively (dicts or stringified dicts/lists),
        reconstruct FeatureSpec[] if present, and skip incomplete rows.
        """
        db = BacktestSQLHelper(read_only=True)
        top_rows = db.select_top_n_strategies_across_studies(top_n=top_n)
        existing = (
            db.get_existing_scored_combinations()
        )  # set of (StrategyConfig_ID, pair, timeframe)

        def _is_mabrouk(row: dict) -> bool:
            k = row.get("StrategyKey") or row.get("strategy_key")
            return str(k).upper() == self.key

        def _as_dict(v: Any) -> dict:
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                v = v.strip()
                if v.startswith("{") and v.endswith("}"):
                    try:
                        return ast.literal_eval(v)
                    except Exception:
                        return {}
            return {}

        def _as_list(v: Any) -> list:
            if isinstance(v, list):
                return v
            if isinstance(v, str):
                v = v.strip()
                if (v.startswith("[") and v.endswith("]")) or (
                    v.startswith("(") and v.endswith(")")
                ):
                    try:
                        return list(ast.literal_eval(v))
                    except Exception:
                        return []
            return []

        def _rebuild_feature_specs(raw) -> list | None:
            feats = _as_list(raw)
            if not feats or FeatureSpec is None:
                return None
            out = []
            for item in feats:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if name in LOOKAHEAD_EXCLUDED_FEATURES:
                    continue
                params = _as_dict(item.get("params", {}))
                prefix = item.get("prefix", name)
                if name:
                    out.append(FeatureSpec(name=name, params=params, prefix=prefix))
            return out or None

        jobs: List[tuple] = []
        for row in top_rows:
            if not _is_mabrouk(row):
                continue

            params = row.get("Parameters") or {}
            execution_cfg = extract_execution_config_from_parameters(
                params, row.get("Timeframe")
            )
            model_key = params.get("model_key")
            model_params = _as_dict(params.get("model_params"))
            thresholds = _as_dict(params.get("thresholds"))
            risk = _as_dict(params.get("risk"))
            raw_feature_specs = params.get("feature_specs")
            feature_specs = _rebuild_feature_specs(raw_feature_specs)

            raw_feature_list = _as_list(raw_feature_specs)
            has_excluded_feature = any(
                isinstance(item, dict)
                and item.get("name") in LOOKAHEAD_EXCLUDED_FEATURES
                for item in raw_feature_list
            )

            if not isinstance(model_key, str) or not thresholds or not risk:
                continue
            if has_excluded_feature:
                continue
            if raw_feature_specs and not feature_specs:
                continue

            for pair in pairs:
                for timeframe in timeframes:
                    strategy_obj = strategy_core.create_strategy_from_kwargs(
                        MabroukAdapter.key,
                        forex_pair=pair,
                        timeframe=timeframe,
                        feature_specs=feature_specs,  # None → defaults inside the strategy
                        model_key=model_key,
                        model_params=model_params,
                        thresholds=thresholds,
                        risk=risk,
                        same_bar_flip_entry=False,
                    )
                    apply_execution_config_to_strategy(
                        strategy_obj, execution_cfg, timeframe=timeframe
                    )
                    resolved_strategy_id = db.select_strategy_configuration(
                        strategy_obj.NAME,
                        strategy_obj.DESCRIPTION,
                        strategy_obj.PARAMETER_SETTINGS,
                    )
                    if resolved_strategy_id is not None and (
                        resolved_strategy_id,
                        pair,
                        timeframe,
                    ) in existing:
                        continue
                    jobs.append((pair, timeframe, strategy_obj))

        return jobs


# Register on import so main.py picks it up via ADAPTER_REGISTRY
register_adapter(MabroukAdapter())

