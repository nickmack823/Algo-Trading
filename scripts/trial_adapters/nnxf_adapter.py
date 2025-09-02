from __future__ import annotations

import ast
import multiprocessing as mp
import re

import optuna

from scripts.data.sql import BacktestSQLHelper
from scripts.indicators.indicator_configs import (
    IndicatorConfig,
    atr_indicators,
    baseline_indicators,
    momentum_indicators,
    trend_indicators,
    volume_indicators,
)
from scripts.strategies import strategy_core

from .base import StrategyTrialAdapter, register_adapter, run_objective_common


def build_indicator_config_for_trial(
    trial: optuna.Trial,
    role: str,
    pool: list[IndicatorConfig],
    use_default_params: bool,
) -> IndicatorConfig:
    name_pool = [c["name"] for c in pool]
    selected_name = trial.suggest_categorical(role, name_pool)
    cfg = next(c for c in pool if c["name"] == selected_name)
    if use_default_params:
        params = cfg["parameters"]
    else:
        params = {}
        for pname, space in cfg.get("parameter_space", {}).items():
            key = f"{role}.{selected_name}.{pname}"
            params[pname] = trial.suggest_categorical(key, space)
    return {
        "name": selected_name,
        "function": cfg["function"],
        "signal_function": cfg["signal_function"],
        "raw_function": cfg["raw_function"],
        "description": cfg.get("description", ""),
        "parameters": params,
    }


class NNFXAdapter(StrategyTrialAdapter):
    key = "NNFX"

    def objective(
        self,
        trial: optuna.Trial,
        forex_pair: str,
        timeframe: str,
        db_queue: mp.Queue,
        ack_dict: dict,
        context: dict,
    ) -> float:
        allowed: dict = context["allowed_indicators"]
        pools: dict[str, list[IndicatorConfig]] = context["indicator_config_spaces"]
        use_defaults = all(
            isinstance(v, (list, tuple)) and len(v) == 1 and v[0] == "all"
            for v in allowed.values()
        )

        atr = build_indicator_config_for_trial(trial, "ATR", pools["ATR"], use_defaults)
        baseline = build_indicator_config_for_trial(
            trial, "Baseline", pools["Baseline"], use_defaults
        )
        c1 = build_indicator_config_for_trial(trial, "C1", pools["C1"], use_defaults)
        c2 = build_indicator_config_for_trial(trial, "C2", pools["C2"], use_defaults)
        volume = build_indicator_config_for_trial(
            trial, "Volume", pools["Volume"], use_defaults
        )
        exit_cfg = build_indicator_config_for_trial(
            trial, "Exit", pools["Exit"], use_defaults
        )

        if c1["name"] == c2["name"] or c1["function"] == c2["function"]:
            raise optuna.exceptions.TrialPruned()

        def build_strategy():
            return strategy_core.create_strategy_from_kwargs(
                "NNFX",
                atr=atr,
                baseline=baseline,
                c1=c1,
                c2=c2,
                volume=volume,
                exit_indicator=exit_cfg,
                forex_pair=forex_pair,
                timeframe=timeframe,
            )

        return run_objective_common(
            build_strategy, trial, forex_pair, timeframe, db_queue, ack_dict
        )

    # Phase builders (unchanged logic from your current main.py)
    def build_phase1_args(
        self,
        pairs,
        timeframes,
        seeds,
        trials_by_timeframe,
        exploration_space="default",
        phase_name="phase1",
    ):
        from scripts.indicators.indicator_configs import (
            atr_indicators,
            baseline_indicators,
            momentum_indicators,
            trend_indicators,
            volume_indicators,
        )

        allowed = {
            "ATR": ["all"],
            "Baseline": ["all"],
            "C1": ["all"],
            "C2": ["all"],
            "Volume": ["all"],
            "Exit": ["all"],
        }
        pools = {
            "ATR": atr_indicators,
            "Baseline": baseline_indicators,
            "C1": trend_indicators + momentum_indicators,
            "C2": trend_indicators + momentum_indicators,
            "Volume": volume_indicators,
            "Exit": trend_indicators + momentum_indicators,
        }
        args = []
        for pair in pairs:
            for tf in timeframes:
                n = trials_by_timeframe[tf]
                for seed in seeds:
                    context = {
                        "allowed_indicators": allowed,
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
        from scripts.data.sql import BacktestSQLHelper

        db = BacktestSQLHelper(read_only=True)
        top_strats = db.select_top_percent_strategies("default", top_percent)

        def _filter(rows, key):
            out = []
            for r in rows:
                k = r.get("StrategyKey") or r.get("strategy_key")
                if k == key:
                    out.append(r)
            return out or rows

        top_strats = _filter(top_strats, self.key)

        role_to_names = {
            k: set() for k in ["ATR", "Baseline", "C1", "C2", "Volume", "Exit"]
        }
        for row in top_strats:
            params = row["Parameters"]
            for role in role_to_names:
                s = params.get(role.lower(), "")
                name = s.split("_")[0]
                if name:
                    role_to_names[role].add(name)

        def filt(pool, names):
            return [c for c in pool if c["name"] in names]

        pools = {
            "ATR": filt(atr_indicators, role_to_names["ATR"]),
            "Baseline": filt(baseline_indicators, role_to_names["Baseline"]),
            "C1": filt(trend_indicators + momentum_indicators, role_to_names["C1"]),
            "C2": filt(trend_indicators + momentum_indicators, role_to_names["C2"]),
            "Volume": filt(volume_indicators, role_to_names["Volume"]),
            "Exit": filt(trend_indicators + momentum_indicators, role_to_names["Exit"]),
        }
        allowed = {role: sorted(list(names)) for role, names in role_to_names.items()}

        args = []
        for pair in pairs:
            for tf in timeframes:
                n = trials_by_timeframe[tf]
                for seed in seeds:
                    context = {
                        "allowed_indicators": allowed,
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

    def build_phase3_args(
        self,
        pairs: list[str],
        timeframes: list[str],
        top_n: int,
    ) -> list[tuple]:
        """
        Exact parity with original Phase-3:
        - select_top_n_strategies_across_studies
        - skip using get_existing_scored_combinations()
        - rebuild IndicatorConfigs from stored parameter strings
        """

        def parse_indicator_field(value: str):
            # "NAME_{...}" → ("NAME", "{...}" -> dict)
            match = re.match(r"^(.*?)_\{(.*)\}$", value)
            if not match:
                return value, {}
            name = match.group(1)
            params_str = match.group(2)
            try:
                params = ast.literal_eval("{" + params_str + "}")
            except Exception:
                params = {}
            return name, params

        key_map = {
            "atr": "ATR",
            "baseline": "Baseline",
            "c1": "C1",
            "c2": "C2",
            "volume": "Volume",
            "exit": "Exit",
        }

        db = BacktestSQLHelper(read_only=True)
        top_rows = db.select_top_n_strategies_across_studies(top_n=top_n)
        existing_combos = (
            db.get_existing_scored_combinations()
        )  # set of (StrategyConfig_ID, pair, timeframe)

        # pools by role
        pools = {
            "ATR": atr_indicators,
            "Baseline": baseline_indicators,
            "C1": trend_indicators + momentum_indicators,
            "C2": trend_indicators + momentum_indicators,
            "Volume": volume_indicators,
            "Exit": trend_indicators + momentum_indicators,
        }

        args: list[tuple] = []

        for strategy_row in top_rows:
            raw_params = strategy_row["Parameters"]
            role_configs: dict[str, dict] = {}

            # Rebuild IndicatorConfigs
            for raw_key, full_str in raw_params.items():
                role = key_map.get(raw_key)
                if not role:
                    continue
                name, parsed_params = parse_indicator_field(full_str)

                matches = [cfg for cfg in pools[role] if cfg["name"] == name]
                if not matches:
                    role_configs = {}
                    break
                cfg = dict(matches[0])  # copy
                cfg["parameters"] = parsed_params
                role_configs[role] = cfg

            if len(role_configs) != 6:
                continue

            for pair in pairs:
                for timeframe in timeframes:
                    combo = (strategy_row["StrategyConfig_ID"], pair, timeframe)
                    if combo in existing_combos:  # exact original skip rule
                        continue
                    strategy_obj = strategy_core.create_strategy_from_kwargs(
                        "NNFX",
                        atr=role_configs["ATR"],
                        baseline=role_configs["Baseline"],
                        c1=role_configs["C1"],
                        c2=role_configs["C2"],
                        volume=role_configs["Volume"],
                        exit_indicator=role_configs["Exit"],
                        forex_pair=pair,
                        timeframe=timeframe,
                    )
                    args.append((pair, timeframe, strategy_obj))
        return args


# register on import
register_adapter(NNFXAdapter())
