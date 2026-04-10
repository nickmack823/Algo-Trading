from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from typing import Callable, Dict

import optuna
import pandas as pd

from scripts.backtester import Backtester
from scripts.config import (
    DEFAULT_SESSION_TEMPLATE_BY_TIMEFRAME,
    DATA_FOLDER,
    END_DATE,
    INTRABAR_MODE_PHASE_1_AND_2,
    MIN_TRADES_PER_DAY,
    PRUNE_THRESHOLD_FACTOR,
    SESSION_TEMPLATE_SEARCH_SPACE_BY_TIMEFRAME,
    TIMEFRAME_DATE_RANGES_PHASE_1_AND_2,
)
from scripts.data.sql import BacktestSQLHelper, HistoricalDataSQLHelper
from scripts.strategies import strategy_core
from scripts.trading_hours import (
    default_execution_config_for_timeframe,
    sanitize_execution_config,
)


def _resolve_date_window(
    timeframe: str, context: dict, default_from: str, default_to: str
) -> tuple[str, str]:
    """
    Support two forms in context['date_window']:
      - {"absolute": {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}}
      - {"relative": {"years": int, "months": int, "days": int}, "anchor_to_config_end": bool}
    Falls back to (default_from, default_to) if nothing provided.
    """
    dw = (context or {}).get("date_window")
    if not dw:
        return default_from, default_to

    # absolute override
    abs_win = dw.get("absolute")
    if abs_win and abs_win.get("from") and abs_win.get("to"):
        return abs_win["from"], abs_win["to"]

    # relative override
    rel = dw.get("relative")
    if rel:
        anchor_to_conf = bool(dw.get("anchor_to_config_end"))
        anchor_to = pd.to_datetime(END_DATE if anchor_to_conf else default_to)
        years = int(rel.get("years", 0))
        months = int(rel.get("months", 0))
        days = int(rel.get("days", 0))
        delta = pd.DateOffset(years=years, months=months, days=days)
        frm = (anchor_to - delta).strftime("%Y-%m-%d")
        to_ = anchor_to.strftime("%Y-%m-%d")
        return frm, to_

    return default_from, default_to


def _resolve_evaluation_window(
    context: dict | None, default_from: str, default_to: str
) -> tuple[str, str]:
    """
    Resolve the score/evaluation window.
    Priority:
      1) context['evaluation_window']
      2) context['date_window'] (same parsing rules as _resolve_date_window)
      3) timeframe default window
    """
    ew = (context or {}).get("evaluation_window") or {}
    if ew.get("from") and ew.get("to"):
        return ew["from"], ew["to"]

    dw = (context or {}).get("date_window") or {}
    abs_win = dw.get("absolute")
    if abs_win and abs_win.get("from") and abs_win.get("to"):
        return abs_win["from"], abs_win["to"]

    rel = dw.get("relative")
    if rel:
        anchor_to_conf = bool(dw.get("anchor_to_config_end"))
        anchor_to = pd.to_datetime(END_DATE if anchor_to_conf else default_to)
        years = int(rel.get("years", 0))
        months = int(rel.get("months", 0))
        days = int(rel.get("days", 0))
        delta = pd.DateOffset(years=years, months=months, days=days)
        frm = (anchor_to - delta).strftime("%Y-%m-%d")
        to_ = anchor_to.strftime("%Y-%m-%d")
        return frm, to_

    return default_from, default_to


EXECUTION_CONFIG_KEY = "__execution__"


def _dedupe_preserve_order(values: list) -> list:
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _resolve_execution_config_for_trial(
    trial: optuna.Trial,
    timeframe: str,
    context: dict | None,
) -> dict:
    session_ctx = (context or {}).get("session") or {}

    # Hard disable hook if needed in context.
    if session_ctx.get("enabled") is False:
        return sanitize_execution_config(
            {
                "session_template": "none",
                "session_timezone": session_ctx.get("session_timezone"),
                "entry_session_only": session_ctx.get("entry_session_only", True),
            },
            timeframe,
        )

    default_cfg = default_execution_config_for_timeframe(timeframe)

    # Optional explicit template override from context.
    explicit_template = session_ctx.get("session_template")

    # Optional list override from context, else timeframe defaults.
    options = session_ctx.get("template_options")
    if not isinstance(options, (list, tuple)) or not options:
        options = SESSION_TEMPLATE_SEARCH_SPACE_BY_TIMEFRAME.get(
            timeframe,
            [DEFAULT_SESSION_TEMPLATE_BY_TIMEFRAME.get(timeframe, "full_24x5")],
        )
    options = _dedupe_preserve_order([str(v) for v in options if v is not None])
    if not options:
        options = [default_cfg["session_template"]]

    if explicit_template:
        selected_template = str(explicit_template)
    elif len(options) == 1:
        selected_template = options[0]
    else:
        selected_template = trial.suggest_categorical(
            "Execution.session_template", options
        )

    resolved = sanitize_execution_config(
        {
            "session_template": selected_template,
            "session_timezone": session_ctx.get("session_timezone")
            or default_cfg.get("session_timezone"),
            "entry_session_only": session_ctx.get(
                "entry_session_only", default_cfg.get("entry_session_only", True)
            ),
        },
        timeframe,
    )
    return resolved


def apply_execution_config_to_strategy(
    strategy: "strategy_core.BaseStrategy",
    execution_config: dict | None,
    *,
    timeframe: str | None = None,
) -> dict:
    resolved = sanitize_execution_config(execution_config, timeframe or strategy.TIMEFRAME)
    params = dict(getattr(strategy, "PARAMETER_SETTINGS", {}) or {})
    params[EXECUTION_CONFIG_KEY] = {
        "session_template": resolved.get("session_template"),
        "session_timezone": resolved.get("session_timezone"),
        "entry_session_only": bool(resolved.get("entry_session_only", True)),
    }
    strategy.PARAMETER_SETTINGS = strategy_core.canonicalize_params(params)
    return resolved


def extract_execution_config_from_parameters(
    parameters: dict | None, timeframe: str | None
) -> dict:
    if isinstance(parameters, dict):
        if isinstance(parameters.get("parameters"), dict):
            parameters = parameters["parameters"]
        raw = parameters.get(EXECUTION_CONFIG_KEY)
        if isinstance(raw, dict):
            return sanitize_execution_config(raw, timeframe)
    return default_execution_config_for_timeframe(timeframe)


# ---- simple dataclass substitute for cross-process acks
class Acknowledgement:
    def __init__(self, ok: bool, payload: dict | None = None, error: str | None = None):
        self.ok = ok
        self.payload = payload
        self.error = error


ACK_TIMEOUT_SECONDS = 30.0
ACK_POLL_SECONDS = 0.02


def build_ack_id(forex_pair: str, timeframe: str, trial_number: int) -> str:
    pair = str(forex_pair).replace("/", "")
    return (
        f"{pair}_{timeframe}_{trial_number}_"
        f"{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    )


def wait_for_ack(
    ack_dict: dict,
    ack_id: str,
    timeout: float = ACK_TIMEOUT_SECONDS,
    poll: float = ACK_POLL_SECONDS,
) -> Acknowledgement:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ack = ack_dict.get(ack_id)
        if ack is not None:
            try:
                del ack_dict[ack_id]
            except KeyError:
                pass
            return ack
        time.sleep(poll)
    return Acknowledgement(ok=False, error=f"ACK timeout for {ack_id}")


# ========== Registry ==========
ADAPTER_REGISTRY: Dict[str, "StrategyTrialAdapter"] = {}


def register_adapter(adapter: "StrategyTrialAdapter") -> None:
    ADAPTER_REGISTRY[adapter.key] = adapter


class StrategyTrialAdapter(ABC):
    key: str  # short name, e.g., "NNFX", "CANDLESTICK"

    @abstractmethod
    def objective(
        self,
        trial: optuna.Trial,
        forex_pair: str,
        timeframe: str,
        db_queue: mp.Queue,
        ack_dict: dict,
        context: dict,
    ) -> float: ...

    # Return tuples used by run_all_studies → run_study_wrapper
    def build_phase1_args(
        self,
        pairs,
        timeframes,
        seeds,
        trials_by_timeframe,
        exploration_space="default",
        phase_name="phase1",
    ):
        return []

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
        return []

    def build_phase3_args(self, pairs, timeframes, top_n: int):
        return []


# ========== Shared trial lifecycle (exactly what main.py used) ==========
def run_objective_common(
    build_strategy: Callable[[], "strategy_core.BaseStrategy"],
    trial: optuna.Trial,
    forex_pair: str,
    timeframe: str,
    db_queue: mp.Queue,
    ack_dict: dict,
    context: dict | None = None,
) -> float:
    def log_error(msg: str):
        logging.error(msg)

    backtest_sqlhelper = BacktestSQLHelper(read_only=True)
    strategy = build_strategy()
    execution_config = _resolve_execution_config_for_trial(trial, timeframe, context)
    execution_config = apply_execution_config_to_strategy(
        strategy, execution_config, timeframe=timeframe
    )
    trial.set_user_attr(
        "execution_session_template", execution_config.get("session_template")
    )

    # ensure strategy_config row exists (writer ACK)
    strategy_config_id = backtest_sqlhelper.select_strategy_configuration(
        strategy.NAME, strategy.DESCRIPTION, strategy.PARAMETER_SETTINGS
    )
    if strategy_config_id is None:
        ack_id = build_ack_id(forex_pair, timeframe, trial.number)
        db_queue.put(
            {
                "purpose": "strategy_config",
                "strategy_name": strategy.NAME,
                "strategy_description": strategy.DESCRIPTION,
                "strategy_parameters": strategy.PARAMETER_SETTINGS,
                "ack_id": ack_id,
            }
        )
        ack = wait_for_ack(ack_dict, ack_id)
        if ack.ok and ack.payload:
            strategy_config_id = ack.payload.get("strategy_config_id")

        # Fallback safety: if ACK is delayed/lost but writer inserted successfully,
        # continue using DB truth rather than failing the whole study.
        if strategy_config_id is None:
            strategy_config_id = backtest_sqlhelper.select_strategy_configuration(
                strategy.NAME, strategy.DESCRIPTION, strategy.PARAMETER_SETTINGS
            )

        if strategy_config_id is None:
            raise RuntimeError(f"Strategy config insert failed: {ack.error}")

    # date window
    tf_rng = TIMEFRAME_DATE_RANGES_PHASE_1_AND_2[timeframe]
    default_from, default_to = tf_rng["from_date"], tf_rng["to_date"]
    from_date, to_date = _resolve_date_window(
        timeframe, context, default_from, default_to
    )
    eval_from, eval_to = _resolve_evaluation_window(context, default_from, default_to)

    # load OHLCV
    data_sqlhelper = HistoricalDataSQLHelper(
        f"{DATA_FOLDER}/{forex_pair.replace('/', '')}.db", read_only=True
    )
    data = data_sqlhelper.get_historical_data(
        table=timeframe, from_date=from_date, to_date=to_date
    )

    # indicators + cache
    strategy.prepare_data(data)
    cache_items = strategy.get_cache_jobs()
    if cache_items:
        db_queue.put({"purpose": "indicator_cache", "cache_items": cache_items})

    backtester = Backtester(
        strategy,
        forex_pair,
        timeframe,
        data,
        initial_balance=10_000,
        metrics_start_date=eval_from,
        metrics_end_date=eval_to,
        intrabar_mode=INTRABAR_MODE_PHASE_1_AND_2,
        session_config=execution_config,
    )

    pair_id = None
    run_metrics_df = None
    score = None
    manually_pruned = False
    ran_new_backtest = False
    calculated_score = False

    def prune_trial(reason: str):
        nonlocal manually_pruned
        manually_pruned = True
        trial.set_user_attr("prune_reason", reason)
        # Don't need to log, Optuna will
        # logging.info(
        #     f"[PRUNED] Trial {trial.number} | {forex_pair} {timeframe} | {reason}"
        # )
        raise optuna.exceptions.TrialPruned(reason)

    try:
        # dedupe
        pair_id = backtest_sqlhelper.select_forex_pair_id(forex_pair)
        run_id = backtest_sqlhelper.select_backtest_run_by_config(
            pair_id, strategy_config_id, timeframe, eval_from, eval_to
        )

        if run_id is not None:
            score = backtest_sqlhelper.select_composite_score(run_id)
            if score is not None:
                prune_trial(
                    f"Duplicate configuration already scored in DB (run_id={run_id}, score={score})."
                )
            else:
                run_metrics_df = backtest_sqlhelper.select_backtest_run_metrics_by_id(
                    run_id
                )
                backtester.set_metrics_df(run_metrics_df)
        else:
            try:
                backtester.run_backtest()
            except Exception as e:
                log_error(
                    f"[{forex_pair}-{timeframe}] (Strategy Config ID: {strategy.CONFIG_ID}) Backtest failed with parameters {strategy.PARAMETER_SETTINGS}: {e}. PRUNING trial."
                )
                prune_trial(
                    f"Backtest execution failed for strategy config {strategy.CONFIG_ID}: {e}"
                )

            # Guardrail: if a test-window start is provided, reject any trade before it.
            eval_from_ts = pd.to_datetime(eval_from)
            has_pretest_trade = any(
                pd.to_datetime(t.entry_timestamp) < eval_from_ts
                for t in backtester.all_trades
            )
            if has_pretest_trade:
                first_trade = min(
                    pd.to_datetime(t.entry_timestamp) for t in backtester.all_trades
                )
                prune_trial(
                    "Pre-test trade guard triggered: "
                    f"first trade {first_trade} < evaluation_window.from {eval_from_ts}."
                )

            run_metrics_df = backtester.get_metrics_df()
            ran_new_backtest = True

        # close RO handle before prune/score
        backtest_sqlhelper.close_connection()
        backtest_sqlhelper = None

        prune_threshold = round(
            MIN_TRADES_PER_DAY[timeframe] * PRUNE_THRESHOLD_FACTOR, 2
        )
        trades_per_day = backtester.get_metric("Trades_Per_Day")
        if trades_per_day < prune_threshold:
            prune_trial(
                f"Insufficient activity: Trades_Per_Day={trades_per_day} < prune_threshold={prune_threshold}."
            )

        # score
        score = backtester.calculate_composite_score()
        calculated_score = True
        if score is None or (score != score) or score in (float("inf"), float("-inf")):
            score = None
            prune_trial("Composite score is invalid (None/NaN/Inf).")

        return score

    except Exception as e:
        if manually_pruned:
            raise e
        logging.error(
            f"Fatal error on {forex_pair} {timeframe}:\n{traceback.format_exc()}"
        )
        sys.exit(1)

    finally:
        if backtest_sqlhelper is not None:
            backtest_sqlhelper.close_connection()

        # enqueue only when new run OR reused metrics + computed score
        if ran_new_backtest or (not ran_new_backtest and calculated_score):
            trial_id = trial.number
            trial_start = (
                trial.datetime_start.isoformat() if trial.datetime_start else None
            )
            study_name = trial.study.study_name
            space_index = study_name.find("space-")
            underscore_index = study_name.find("_seed")
            exploration_space = (
                study_name[space_index + 6 : underscore_index]
                if space_index != -1 and underscore_index != -1
                else ""
            )
            db_queue.put(
                {
                    "pair_id": pair_id,
                    "strategy_name": strategy.NAME,
                    "strategy_description": strategy.DESCRIPTION,
                    "strategy_parameters": strategy.PARAMETER_SETTINGS,
                    "timeframe": timeframe,
                    "metrics_df": run_metrics_df,
                    "study_name": study_name,
                    "trial_id": trial_id,
                    "score": score,
                    "exploration_space": exploration_space,
                    "trial_start": trial_start,
                    "purpose": "trial",
                }
            )


# ========== Generic delegator (what main.py called `objective`) ==========
def objective(trial, forex_pair, timeframe, db_queue, ack_dict, strategy_key, context):
    adapter = ADAPTER_REGISTRY[strategy_key]
    return adapter.objective(trial, forex_pair, timeframe, db_queue, ack_dict, context)
