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

from .base_adapter import (
    StrategyTrialAdapter,
    apply_execution_config_to_strategy,
    extract_execution_config_from_parameters,
    register_adapter,
    run_objective_common,
)


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

# Family map used for diversified sampling.
# Goal: avoid over-concentrating on highly redundant pattern subsets.
PATTERN_FAMILY_BY_NAME: dict[str, str] = {
    # Indecision / neutral
    "CANDLE_DOJI": "indecision",
    "CANDLE_DOJISTAR": "indecision",
    "CANDLE_LONGLEGGEDDOJI": "indecision",
    "CANDLE_RICKSHAWMAN": "indecision",
    "CANDLE_HIGHWAVE": "indecision",
    "CANDLE_SPINNINGTOP": "indecision",
    "CANDLE_SHORTLINE": "indecision",
    # Bullish reversal
    "CANDLE_HAMMER": "bullish_reversal",
    "CANDLE_INVERTEDHAMMER": "bullish_reversal",
    "CANDLE_TAKURI": "bullish_reversal",
    "CANDLE_PIERCING": "bullish_reversal",
    "CANDLE_MORNINGSTAR": "bullish_reversal",
    "CANDLE_MORNINGDOJISTAR": "bullish_reversal",
    "CANDLE_3WHITESOLDIERS": "bullish_reversal",
    "CANDLE_3STARSINSOUTH": "bullish_reversal",
    "CANDLE_MATCHINGLOW": "bullish_reversal",
    "CANDLE_HOMINGPIGEON": "bullish_reversal",
    "CANDLE_LADDERBOTTOM": "bullish_reversal",
    "CANDLE_UNIQUE3RIVER": "bullish_reversal",
    "CANDLE_STICKSANDWICH": "bullish_reversal",
    "CANDLE_DRAGONFLYDOJI": "bullish_reversal",
    # Bearish reversal
    "CANDLE_HANGINGMAN": "bearish_reversal",
    "CANDLE_SHOOTINGSTAR": "bearish_reversal",
    "CANDLE_DARKCLOUDCOVER": "bearish_reversal",
    "CANDLE_EVENINGSTAR": "bearish_reversal",
    "CANDLE_EVENINGDOJISTAR": "bearish_reversal",
    "CANDLE_3BLACKCROWS": "bearish_reversal",
    "CANDLE_IDENTICAL3CROWS": "bearish_reversal",
    "CANDLE_ADVANCEBLOCK": "bearish_reversal",
    "CANDLE_STALLEDPATTERN": "bearish_reversal",
    "CANDLE_UPSIDEGAP2CROWS": "bearish_reversal",
    "CANDLE_2CROWS": "bearish_reversal",
    "CANDLE_GRAVESTONEDOJI": "bearish_reversal",
    # Dual-sided reversal (direction depends on sign/context)
    "CANDLE_ENGULFING": "reversal_dual",
    "CANDLE_HARAMI": "reversal_dual",
    "CANDLE_HARAMICROSS": "reversal_dual",
    "CANDLE_COUNTERATTACK": "reversal_dual",
    "CANDLE_ABANDONEDBABY": "reversal_dual",
    "CANDLE_BELTHOLD": "reversal_dual",
    "CANDLE_BREAKAWAY": "reversal_dual",
    "CANDLE_KICKING": "reversal_dual",
    "CANDLE_KICKINGBYLENGTH": "reversal_dual",
    "CANDLE_TRISTAR": "reversal_dual",
    "CANDLE_3INSIDE": "reversal_dual",
    "CANDLE_3LINESTRIKE": "reversal_dual",
    "CANDLE_CONCEALBABYSWALL": "reversal_dual",
    # Continuation
    "CANDLE_RISEFALL3METHODS": "continuation",
    "CANDLE_XSIDEGAP3METHODS": "continuation",
    "CANDLE_TASUKIGAP": "continuation",
    "CANDLE_GAPSIDESIDEWHITE": "continuation",
    "CANDLE_SEPARATINGLINES": "continuation",
    "CANDLE_MATHOLD": "continuation",
    "CANDLE_ONNECK": "continuation",
    "CANDLE_INNECK": "continuation",
    "CANDLE_THRUSTING": "continuation",
    # Momentum / body-strength
    "CANDLE_MARUBOZU": "momentum_body",
    "CANDLE_CLOSINGMARUBOZU": "momentum_body",
    "CANDLE_LONGLINE": "momentum_body",
    # Trap / false-break
    "CANDLE_HIKKAKE": "trap_false_break",
    "CANDLE_HIKKAKEMOD": "trap_false_break",
}

_pattern_set = set(ALL_PATTERN_NAMES)
_family_map_set = set(PATTERN_FAMILY_BY_NAME)
_missing_family_map = sorted(_pattern_set - _family_map_set)
_extra_family_map = sorted(_family_map_set - _pattern_set)
if _missing_family_map or _extra_family_map:
    raise RuntimeError(
        "Candlestick family map mismatch. "
        f"missing={_missing_family_map} extra={_extra_family_map}"
    )


def _sample_patterns_family_aware(
    trial: optuna.Trial,
    available_patterns: list[str] | tuple[str, ...],
) -> list[str]:
    pool = tuple(
        sorted({name for name in available_patterns if name in PATTERN_FAMILY_BY_NAME})
    )
    if not pool:
        raise optuna.exceptions.TrialPruned("No valid candlestick patterns in pool.")

    max_patterns = min(6, len(pool))
    min_patterns = 3 if len(pool) >= 3 else 1
    k = trial.suggest_int("Patterns.K", min_patterns, max_patterns)

    family_to_patterns: dict[str, tuple[str, ...]] = {}
    for name in pool:
        family = PATTERN_FAMILY_BY_NAME[name]
        family_to_patterns.setdefault(family, [])
        family_to_patterns[family].append(name)
    family_to_patterns = {
        fam: tuple(sorted(names)) for fam, names in family_to_patterns.items()
    }

    family_pool = tuple(sorted(family_to_patterns.keys()))
    max_family_count = min(k, len(family_pool))
    min_family_count = 1 if max_family_count <= 1 else 2
    family_k = trial.suggest_int(
        "Patterns.Families.K", min_family_count, max_family_count
    )

    fam_sig = _space_signature(list(family_pool))
    sampled_families = [
        trial.suggest_categorical(f"Patterns.Families.pick.{i}.{fam_sig}", family_pool)
        for i in range(8)
    ]

    selected_families: list[str] = []
    seen_families = set()
    for fam in sampled_families:
        if fam in seen_families:
            continue
        selected_families.append(fam)
        seen_families.add(fam)
        if len(selected_families) >= family_k:
            break
    if len(selected_families) < family_k:
        for fam in family_pool:
            if fam in seen_families:
                continue
            selected_families.append(fam)
            seen_families.add(fam)
            if len(selected_families) >= family_k:
                break

    patterns: list[str] = []
    seen_patterns = set()

    # Seed one pattern per selected family.
    for fam in selected_families:
        choices = family_to_patterns[fam]
        fam_choice_sig = _space_signature(list(choices))
        picked = trial.suggest_categorical(
            f"Patterns.Family.{fam}.seed.{fam_choice_sig}", choices
        )
        if picked not in seen_patterns:
            patterns.append(picked)
            seen_patterns.add(picked)
        else:
            for alt in choices:
                if alt not in seen_patterns:
                    patterns.append(alt)
                    seen_patterns.add(alt)
                    break

    # Fill remaining slots from the selected-family union.
    if len(patterns) < k:
        union_pool = tuple(
            sorted(
                {
                    name
                    for fam in selected_families
                    for name in family_to_patterns[fam]
                }
            )
        )
        union_sig = _space_signature(list(union_pool))
        for i in range(12):
            cand = trial.suggest_categorical(
                f"Patterns.extra.pick.{i}.{union_sig}", union_pool
            )
            if cand in seen_patterns:
                continue
            patterns.append(cand)
            seen_patterns.add(cand)
            if len(patterns) >= k:
                break

        if len(patterns) < k:
            for cand in union_pool:
                if cand in seen_patterns:
                    continue
                patterns.append(cand)
                seen_patterns.add(cand)
                if len(patterns) >= k:
                    break

    return patterns


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
            use_trend_filter = trial.suggest_categorical(
                "use_trend_filter", [False, True]
            )
            if use_trend_filter:
                trend = _sample_indicator(trial, "Trend", pools["Trend"], use_defaults)

        available = (
            ALL_PATTERN_NAMES if "all" in allowed["Patterns"] else allowed["Patterns"]
        )
        patterns = _sample_patterns_family_aware(trial, available)

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
            params_for_exec = (
                params.get("parameters")
                if isinstance(params.get("parameters"), dict)
                else params
            )
            execution_cfg = extract_execution_config_from_parameters(
                params_for_exec, row.get("Timeframe")
            )
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
                    ) in existing_combos:
                        continue
                    jobs.append((pair, timeframe, strategy_obj))

        return jobs


register_adapter(CandlestickAdapter())
