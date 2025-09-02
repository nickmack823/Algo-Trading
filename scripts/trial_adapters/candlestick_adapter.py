from __future__ import annotations

import multiprocessing as mp
from typing import List

import optuna

from scripts.indicators.indicator_configs import (
    IndicatorConfig,
    atr_indicators,
    baseline_indicators,
    ta_lib_candlestick,
    trend_indicators,
)
from scripts.strategies import strategy_core

from .base import StrategyTrialAdapter, register_adapter, run_objective_common


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


# Available candlestick pattern names
ALL_PATTERN_NAMES: List[str] = [c["name"] for c in ta_lib_candlestick]


class CandlestickAdapter(StrategyTrialAdapter):
    key = "CANDLESTICK"

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
        pool = tuple(sorted(available))  # deterministic
        k = trial.suggest_int("Patterns.K", 3, min(10, max(3, len(pool))))
        max_start = max(0, len(pool) - k)
        start = trial.suggest_int(
            "Patterns.offset", 0, max_start
        )  # primitive => no warning
        patterns = list(pool[start : start + k])  # full names, not characters

        # thresholds
        min_abs_score = trial.suggest_int("min_abs_score", 40, 80, step=10)
        min_body_atr = trial.suggest_float("min_body_atr", 0.2, 0.6, step=0.1)
        align_with_baseline = trial.suggest_categorical(
            "align_with_baseline", [True, False]
        )
        cooldown_bars = trial.suggest_int("cooldown_bars", 1, 4)
        opposite_exit = trial.suggest_categorical("opposite_exit", [True, False])

        def build_strategy():
            return strategy_core.create_strategy_from_kwargs(
                "CANDLESTICK",
                atr=atr,
                baseline=baseline,
                trend_filter=trend,
                patterns=patterns,
                min_abs_score=min_abs_score,
                min_body_atr=min_body_atr,
                align_with_baseline=align_with_baseline,
                cooldown_bars=cooldown_bars,
                opposite_exit=opposite_exit,
                forex_pair=forex_pair,
                timeframe=timeframe,
            )

        return run_objective_common(
            build_strategy, trial, forex_pair, timeframe, db_queue, ack_dict
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
        # Simplified: reuse Phase1 pools but switch from ["all"] to the top-used names/patterns if you’ve stored them.
        # If your DB stores Candlestick params (ATR/Baseline/Trend/Patterns), reconstruct the name sets here similar to NNFX.
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

    def build_phase3_args(self, pairs, timeframes, top_n: int):
        # Optional: mirror your NNFX Phase-3 “fixed eval” behavior using stored Candlestick parameters.
        return []


register_adapter(CandlestickAdapter())
