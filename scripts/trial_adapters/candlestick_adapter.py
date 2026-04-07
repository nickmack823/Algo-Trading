from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing as mp
import re
from typing import List

import optuna

from scripts.data.sql import BacktestSQLHelper
from scripts.indicators.indicator_configs import (
    IndicatorConfig,
    atr_indicators,
    baseline_indicators,
    ta_lib_candlestick,
    trend_indicators,
)
from scripts.strategies import strategy_core

from .base_adapter import StrategyTrialAdapter, register_adapter, run_objective_common


# Stable signature so Optuna params remain deterministic for a given pool.
def _space_signature(values: list) -> str:
    blob = json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


# Helper: indicator sampling (same pattern as NNFX)
def _sample_indicator(
    trial: optuna.Trial,
    role: str,
    pool: list[IndicatorConfig],
    use_default_params: bool,
) -> IndicatorConfig:
    names = [c["name"] for c in pool]
    selected = trial.suggest_categorical(role, names)
    cfg = next(c for c in pool if c["name"] == selected)
    if use_default_params:
        params = cfg["parameters"]
    else:
        params = {}
        for pname, space in cfg.get("parameter_space", {}).items():
            key = f"{role}.{selected}.{pname}"
            params[pname] = trial.suggest_categorical(key, space)
    return {
        "name": selected,
        "function": cfg["function"],
        "signal_function": cfg.get("signal_function"),
        "raw_function": cfg.get("raw_function"),
        "description": cfg.get("description", ""),
        "parameters": params,
    }


def _parse_indicator_field(value) -> tuple[str | None, dict]:
    if value is None:
        return None, {}
    if not isinstance(value, str):
        return str(value), {}

    value = value.strip()
    if not value:
        return None, {}

    match = re.match(r"^(.*?)_\{(.*)\}$", value)
    if not match:
        return value, {}

    name = match.group(1).strip()
    param_str = "{" + match.group(2) + "}"
    try:
        params = ast.literal_eval(param_str)
    except Exception:
        params = {}
    if not isinstance(params, dict):
        params = {}
    return name or None, params


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "1", "yes", "y"}:
            return True
        if raw in {"false", "0", "no", "n"}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_strategy_key_from_study_name(study_name: str | None) -> str | None:
    if not isinstance(study_name, str):
        return None
    m = re.search(
        r"_([^_]+)_\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}_space-",
        study_name,
    )
    if not m:
        return None
    return m.group(1)


def _build_indicator_from_name(
    pool: list[IndicatorConfig], indicator_name: str | None, params: dict | None
) -> IndicatorConfig | None:
    if not indicator_name:
        return None

    match = next((cfg for cfg in pool if cfg["name"] == indicator_name), None)
    if match is None:
        return None

    return {
        "name": match["name"],
        "function": match["function"],
        "signal_function": match.get("signal_function"),
        "raw_function": match.get("raw_function"),
        "description": match.get("description", ""),
        "parameters": params if isinstance(params, dict) else {},
    }


def _parse_patterns_field(value) -> list[str]:
    if value is None:
        return []

    items: list[str] = []
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, (list, tuple, set)):
        items = [str(x).strip() for x in value if str(x).strip()]

    valid = set(ALL_PATTERN_NAMES)
    out: list[str] = []
    seen = set()
    for name in items:
        if name in valid and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _is_candlestick_row(row: dict, key: str) -> bool:
    parsed_key = _parse_strategy_key_from_study_name(row.get("Study_Name"))
    if parsed_key is not None:
        return parsed_key == key

    params = row.get("Parameters")
    if not isinstance(params, dict):
        return False

    if "parameters" in params and isinstance(params["parameters"], dict):
        params = params["parameters"]

    # Fallback heuristic if Study_Name format changes.
    return "patterns" in params and "atr" in params and "baseline" in params


# Available candlestick pattern names
ALL_PATTERN_NAMES: List[str] = [c["name"] for c in ta_lib_candlestick]


class CandlestickAdapter(StrategyTrialAdapter):
    key = "Candlestick"

    def objective(
        self,
        trial: optuna.Trial,
        forex_pair: str,
        timeframe: str,
        db_queue: mp.Queue,
        ack_dict: dict,
        context: dict,
    ) -> float:
        # context provides pools + allowed names just like NNFX
        pools = context[
            "indicator_config_spaces"
        ]  # {"ATR": [...], "Baseline": [...], "Trend": [...]}
        allowed = context[
            "allowed"
        ]  # {"ATR": ["all"]|[names], "Baseline": ..., "Trend": ..., "Patterns": ["all"]|[names]}
        use_defaults = all(v == ["all"] for k, v in allowed.items() if k != "Patterns")

        atr = _sample_indicator(trial, "ATR", pools["ATR"], use_defaults)
        baseline = _sample_indicator(trial, "Baseline", pools["Baseline"], use_defaults)

        trend = None
        if pools.get("Trend"):
            trend = _sample_indicator(trial, "Trend", pools["Trend"], use_defaults)

        available = (
            ALL_PATTERN_NAMES if "all" in allowed["Patterns"] else allowed["Patterns"]
        )
        pool = tuple(sorted({name for name in available if name in ALL_PATTERN_NAMES}))
        if not pool:
            raise optuna.exceptions.TrialPruned("No valid candlestick patterns in pool.")

        max_patterns = min(6, len(pool))
        min_patterns = 3 if len(pool) >= 3 else 1
        k = trial.suggest_int("Patterns.K", min_patterns, max_patterns)
        pool_sig = _space_signature(list(pool))

        raw_picks: list[str] = []
        for i in range(10):
            raw_picks.append(trial.suggest_categorical(f"Patterns.pick.{i}.{pool_sig}", pool))

        patterns: list[str] = []
        seen = set()
        for name in raw_picks:
            if name in seen:
                continue
            patterns.append(name)
            seen.add(name)
            if len(patterns) >= k:
                break

        # Rare case: duplicates reduced cardinality below k; fill deterministically.
        if len(patterns) < k:
            for name in pool:
                if name in seen:
                    continue
                patterns.append(name)
                seen.add(name)
                if len(patterns) >= k:
                    break

        # thresholds
        min_abs_score = trial.suggest_int("min_abs_score", 40, 80, step=10)
        min_body_atr = trial.suggest_float("min_body_atr", 0.2, 0.6, step=0.1)
        align_with_baseline = trial.suggest_categorical(
            "align_with_baseline", [True, False]
        )
        cooldown_bars = trial.suggest_int("cooldown_bars", 1, 4)
        opposite_exit = trial.suggest_categorical("opposite_exit", [True, False])
        min_votes = trial.suggest_int("min_votes", 1, 4)
        dominance_ratio = trial.suggest_float("dominance_ratio", 1.0, 1.8, step=0.1)
        exit_min_votes = trial.suggest_int("exit_min_votes", 1, 4)

        def build_strategy():
            return strategy_core.create_strategy_from_kwargs(
                self.key,
                atr=atr,
                baseline=baseline,
                trend_filter=trend,
                patterns=patterns,
                min_abs_score=min_abs_score,
                min_body_atr=min_body_atr,
                align_with_baseline=align_with_baseline,
                cooldown_bars=cooldown_bars,
                opposite_exit=opposite_exit,
                min_votes=min_votes,
                dominance_ratio=dominance_ratio,
                exit_min_votes=exit_min_votes,
                forex_pair=forex_pair,
                timeframe=timeframe,
            )

        return run_objective_common(
            build_strategy,
            trial,
            forex_pair,
            timeframe,
            db_queue,
            ack_dict,
            context,
        )

    def build_phase1_args(
        self,
        pairs,
        timeframes,
        seeds,
        trials_by_timeframe,
        exploration_space="default",
        phase_name="phase1",
    ):
        allowed = {
            "ATR": ["all"],
            "Baseline": ["all"],
            "Trend": ["all"],
            "Patterns": ["all"],
        }
        pools = {
            "ATR": atr_indicators,
            "Baseline": baseline_indicators,
            "Trend": trend_indicators,
        }

        args = []
        for pair in pairs:
            for tf in timeframes:
                n = trials_by_timeframe[tf]
                for seed in seeds:
                    context = {
                        "allowed": allowed,
                        "indicator_config_spaces": pools,
                        "exploration_space": exploration_space,
                    }
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
        db = BacktestSQLHelper(read_only=True)
        try:
            top_rows = db.select_top_percent_strategies("default", top_percent)
        finally:
            db.close_connection()

        top_rows = [r for r in top_rows if _is_candlestick_row(r, self.key)]

        role_to_names = {
            "ATR": set(),
            "Baseline": set(),
            "Trend": set(),
            "Patterns": set(),
        }
        for row in top_rows:
            params = row.get("Parameters")
            if not isinstance(params, dict):
                continue
            if "parameters" in params and isinstance(params["parameters"], dict):
                params = params["parameters"]

            atr_name, _ = _parse_indicator_field(params.get("atr"))
            baseline_name, _ = _parse_indicator_field(params.get("baseline"))
            trend_name, _ = _parse_indicator_field(params.get("trend"))
            pattern_names = _parse_patterns_field(params.get("patterns"))

            if atr_name:
                role_to_names["ATR"].add(atr_name)
            if baseline_name:
                role_to_names["Baseline"].add(baseline_name)
            if trend_name and trend_name.lower() not in {"none", "null"}:
                role_to_names["Trend"].add(trend_name)
            for p in pattern_names:
                role_to_names["Patterns"].add(p)

        def _filter_pool(pool, names: set[str]) -> list[IndicatorConfig]:
            if not names:
                return []
            return [cfg for cfg in pool if cfg["name"] in names]

        filtered_atr = _filter_pool(atr_indicators, role_to_names["ATR"])
        filtered_baseline = _filter_pool(baseline_indicators, role_to_names["Baseline"])
        filtered_trend = _filter_pool(trend_indicators, role_to_names["Trend"])

        pools = {
            "ATR": filtered_atr or atr_indicators,
            "Baseline": filtered_baseline or baseline_indicators,
            # Keep empty to explicitly allow "no trend filter" in narrowed phase2 when no trend names survived.
            "Trend": filtered_trend,
        }

        allowed = {
            "ATR": sorted(role_to_names["ATR"]) if role_to_names["ATR"] else ["all"],
            "Baseline": (
                sorted(role_to_names["Baseline"])
                if role_to_names["Baseline"]
                else ["all"]
            ),
            "Trend": sorted(role_to_names["Trend"]),
            "Patterns": (
                sorted(role_to_names["Patterns"])
                if role_to_names["Patterns"]
                else ["all"]
            ),
        }

        args = []
        for pair in pairs:
            for tf in timeframes:
                n = trials_by_timeframe[tf]
                for seed in seeds:
                    context = {
                        "allowed": allowed,
                        "indicator_config_spaces": pools,
                        "exploration_space": exploration_space,
                    }
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

    def build_phase3_args(self, pairs, timeframes, top_n: int):
        db = BacktestSQLHelper(read_only=True)
        try:
            top_rows = db.select_top_n_strategies_across_studies(top_n=top_n)
            existing_combos = db.get_existing_scored_combinations()
        finally:
            db.close_connection()

        jobs: list[tuple] = []
        for row in top_rows:
            if not _is_candlestick_row(row, self.key):
                continue

            params = row.get("Parameters")
            if not isinstance(params, dict):
                continue
            if "parameters" in params and isinstance(params["parameters"], dict):
                params = params["parameters"]

            atr_name, atr_params = _parse_indicator_field(params.get("atr"))
            baseline_name, baseline_params = _parse_indicator_field(params.get("baseline"))
            trend_name, trend_params = _parse_indicator_field(params.get("trend"))
            pattern_names = _parse_patterns_field(params.get("patterns"))

            atr_cfg = _build_indicator_from_name(atr_indicators, atr_name, atr_params)
            baseline_cfg = _build_indicator_from_name(
                baseline_indicators, baseline_name, baseline_params
            )
            trend_cfg = None
            if trend_name and str(trend_name).lower() not in {"none", "null"}:
                trend_cfg = _build_indicator_from_name(
                    trend_indicators,
                    trend_name,
                    trend_params,
                )

            if atr_cfg is None or baseline_cfg is None or not pattern_names:
                continue
            if trend_name and str(trend_name).lower() not in {"none", "null"} and trend_cfg is None:
                continue

            min_abs_score = _coerce_int(params.get("min_abs_score"), 50)
            min_body_atr = _coerce_float(params.get("min_body_atr"), 0.3)
            align_with_baseline = _coerce_bool(
                params.get("align_with_baseline"), True
            )
            cooldown_bars = _coerce_int(params.get("cooldown_bars"), 2)
            opposite_exit = _coerce_bool(params.get("opposite_exit"), True)
            min_votes = _coerce_int(params.get("min_votes"), 2)
            dominance_ratio = _coerce_float(params.get("dominance_ratio"), 1.25)
            exit_min_votes = _coerce_int(params.get("exit_min_votes"), 2)

            for pair in pairs:
                for timeframe in timeframes:
                    combo = (row["StrategyConfig_ID"], pair, timeframe)
                    if combo in existing_combos:
                        continue

                    strategy_obj = strategy_core.create_strategy_from_kwargs(
                        self.key,
                        atr=atr_cfg,
                        baseline=baseline_cfg,
                        trend_filter=trend_cfg,
                        patterns=pattern_names,
                        min_abs_score=min_abs_score,
                        min_body_atr=min_body_atr,
                        align_with_baseline=align_with_baseline,
                        cooldown_bars=cooldown_bars,
                        opposite_exit=opposite_exit,
                        min_votes=min_votes,
                        dominance_ratio=dominance_ratio,
                        exit_min_votes=exit_min_votes,
                        forex_pair=pair,
                        timeframe=timeframe,
                    )
                    jobs.append((pair, timeframe, strategy_obj))

        return jobs


register_adapter(CandlestickAdapter())
