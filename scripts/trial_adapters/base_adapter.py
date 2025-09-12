from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import time
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Dict

import optuna
import pandas as pd

from scripts.backtester import Backtester
from scripts.config import (
    DATA_FOLDER,
    END_DATE,
    MIN_TRADES_PER_DAY,
    PRUNE_THRESHOLD_FACTOR,
    TIMEFRAME_DATE_RANGES_PHASE_1_AND_2,
)
from scripts.data.sql import BacktestSQLHelper, HistoricalDataSQLHelper
from scripts.strategies import strategy_core


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


# ---- simple dataclass substitute for cross-process acks
class Acknowledgement:
    def __init__(self, ok: bool, payload: dict | None = None, error: str | None = None):
        self.ok = ok
        self.payload = payload
        self.error = error


def wait_for_ack(
    ack_dict: dict, ack_id: str, timeout: float = 10.0, poll: float = 0.02
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

    # ensure strategy_config row exists (writer ACK)
    strategy_config_id = backtest_sqlhelper.select_strategy_configuration(
        strategy.NAME, strategy.DESCRIPTION, strategy.PARAMETER_SETTINGS
    )
    if strategy_config_id is None:
        ack_id = f"{forex_pair}_{timeframe}_{trial.number}_{time.time()}"
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
        if not ack.ok:
            raise RuntimeError(f"Strategy config insert failed: {ack.error}")

        strategy_config_id = backtest_sqlhelper.select_strategy_configuration(
            strategy.NAME, strategy.DESCRIPTION, strategy.PARAMETER_SETTINGS
        )

    # date window
    tf_rng = TIMEFRAME_DATE_RANGES_PHASE_1_AND_2[timeframe]
    from_date, to_date = tf_rng["from_date"], tf_rng["to_date"]
    from_date, to_date = _resolve_date_window(timeframe, context, from_date, to_date)

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
        strategy, forex_pair, timeframe, data, initial_balance=10_000
    )

    pair_id = None
    run_metrics_df = None
    score = None
    manually_pruned = False
    ran_new_backtest = False
    calculated_score = False

    try:
        # dedupe
        pair_id = backtest_sqlhelper.select_forex_pair_id(forex_pair)
        run_id = backtest_sqlhelper.select_backtest_run_by_config(
            pair_id, strategy_config_id, timeframe, from_date, to_date
        )

        if run_id is not None:
            score = backtest_sqlhelper.select_composite_score(run_id)
            if score is not None:
                manually_pruned = True
                raise optuna.exceptions.TrialPruned()
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
                manually_pruned = True
                raise optuna.exceptions.TrialPruned()
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
            manually_pruned = True
            raise optuna.exceptions.TrialPruned()

        # score
        score = backtester.calculate_composite_score()
        calculated_score = True
        if score is None or (score != score) or score in (float("inf"), float("-inf")):
            manually_pruned = True
            score = None
            raise optuna.exceptions.TrialPruned()

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
