import datetime
import gc
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import winsound
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, stdev

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)

from scripts import strategies, utilities
from scripts.backtester import Backtester
from scripts.config import (
    DATA_FOLDER,
    MAJOR_FOREX_PAIRS,
    MIN_TRADES_PER_DAY,
    N_STARTUP_TRIALS_PERCENTAGE,
    OPTUNA_STUDIES_FOLDER,
    PHASE2_TOP_PERCENT,
    PRUNE_THRESHOLD_FACTOR,
    TIMEFRAME_DATE_RANGES,
    TIMEFRAMES,
)
from scripts.indicators import (
    IndicatorConfig,
    atr_indicators,
    baseline_indicators,
    momentum_indicators,
    trend_indicators,
    volume_indicators,
)
from scripts.sql import (
    BacktestSQLHelper,
    HistoricalDataSQLHelper,
    IndicatorCacheSQLHelper,
)

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)

# db_queue = mp.Queue()
# done_flag = mp.Value("b", False)


def log_error(message: str, filename: str = "error_log.txt"):
    """Appends a timestamped error message to a file."""
    logging.error(message)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(log_line)


def test_one():
    strategy = strategies.NNFXStrategy(
        atr=atr_indicators[0],
        baseline=baseline_indicators[0],
        c1=trend_indicators[2],
        c2=trend_indicators[3],
        volume=volume_indicators[-1],
        exit_indicator=trend_indicators[4],
        forex_pair="USDCHF",
        timeframe="4_hour",
    )

    logging.info(f"Current strategy: {strategy}")

    backtester = Backtester(strategy, "USDCHF", "4_hour", initial_balance=10_000)

    # Time the backtest
    start_time = time.time()

    # cProfile.run("backtester.run_backtest()", sort="cumtime")
    backtester.run_backtest()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Backtest completed in {elapsed_time:.2f} seconds")

    backtester.save_run()

    trades = backtester.save_trades()


def wait_for_ack(ack_dict, ack_id, timeout=10):
    start = time.time()
    while True:
        if ack_dict.get(ack_id):
            del ack_dict[ack_id]  # Cleanup
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Ack timeout for {ack_id}")
        time.sleep(0.1)

    return


def run_fixed_strategy_evaluation(
    strategy: strategies.NNFXStrategy,
    pair: str,
    timeframe: str,
    phase_name: str,
    exploration_space: str,
    db_queue: mp.Queue,
):
    sql = BacktestSQLHelper(read_only=True)
    pair_id = sql.select_forex_pair_id(pair)
    strategy_config_id = sql.select_strategy_configuration(
        strategy.NAME, strategy.DESCRIPTION, strategy.PARAMETER_SETTINGS
    )

    from_date, to_date = (
        TIMEFRAME_DATE_RANGES[timeframe]["from_date"],
        TIMEFRAME_DATE_RANGES[timeframe]["to_date"],
    )

    study_name = f"{phase_name}_{pair.replace('/', '')}_{timeframe}_NNFX_{from_date}to{to_date}_space-{exploration_space}"

    data_sqlhelper = HistoricalDataSQLHelper(
        f"{DATA_FOLDER}/{pair.replace('/', '')}.db", read_only=True
    )
    data = data_sqlhelper.get_historical_data(
        table=timeframe, from_date=from_date, to_date=to_date
    )

    strategy.prepare_data(data)
    cache_items = strategy.get_cache_jobs()
    if cache_items:
        db_queue.put({"purpose": "indicator_cache", "cache_items": cache_items})

    backtester = Backtester(strategy, pair, timeframe, data, initial_balance=10_000)

    try:
        backtester.run_backtest()
    except Exception as e:
        log_error(f"[{pair}-{timeframe}] Error in fixed eval: {e}")
        return

    metrics_df = backtester.get_metrics_df()
    score = backtester.calculate_composite_score()
    if score is None or score != score or score in (float("inf"), float("-inf")):
        log_error(f"[{pair}-{timeframe}] Invalid score")
        return

    db_queue.put(
        {
            "purpose": "backtest_run",
            "pair_id": pair_id,
            "strategy_config_id": strategy_config_id,
            "timeframe": timeframe,
            "metrics_df": metrics_df,
            "study_name": study_name,
            "score": score,
            "exploration_space": exploration_space,
        }
    )


def early_stop_callback(N=50, tolerance=0.05, min_trials=30, min_score_threshold=5.0):
    """
    Stop Optuna study early if it meets certain criteria.

    There are two criteria for stopping the study:

    1. If the best score has not improved by at least `tolerance` in the last
       `N` trials, stop the study.
    2. If the best score of the last `N` trials is less than
       `min_score_threshold`, stop the study.

    Additionally, the study will not be stopped until at least `min_trials`
    trials have been completed.

    If the study is stopped due to criterion 1, the stop reason will be
    "no_improvement_{N}_trials". If the study is stopped due to criterion 2, the
    stop reason will be "low_scores_last_{N}_trials".

    Parameters
    ----------
    N : int, optional
        The number of recent trials to consider when stopping the study, by
        default 50.
    tolerance : float, optional
        The minimum improvement in the best score to not consider it a
        "no improvement", by default 0.05 (value based on typical composite score range)
    min_trials : int, optional
        The minimum number of trials that must be completed before the study
        can be stopped, by default 30.
    min_score_threshold : float, optional
        The minimum score that must be exceeded by the best score of the last
        `N` trials, by default 5.0.

    Returns
    -------
    callback
        A callback function that can be passed to Optuna's `Study.optimize`
        method to stop the study early if certain criteria are met.
    """

    def callback(study, trial):
        completed = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if len(completed) < max(N + 10, min_trials):
            return

        best = study.best_trial.value
        recent = [t.value for t in completed[-N:]]

        no_meaningful_improvement = all(v < best - tolerance for v in recent)
        all_bad = max(recent) < min_score_threshold

        if no_meaningful_improvement:
            reason = f"no_improvement_{N}_trials"
            study.set_user_attr("stop_reason", reason)
            logging.info(f"[EARLY STOP] {study.study_name} | Reason: {reason}")
            study.stop()
        elif all_bad:
            reason = f"low_scores_last_{N}_trials"
            study.set_user_attr("stop_reason", reason)
            logging.info(f"[EARLY STOP] {study.study_name} | Reason: {reason}")
            study.stop()

    return callback


def save_optuna_report(study: optuna.Study, output_path: str):
    """Saves Optuna report to disk

    Args:
        study (optuna.Study): The Optuna study
        output_path (str): The output path to save to
    """
    logging.info("Saving Optuna report...")
    os.makedirs(output_path, exist_ok=True)

    plots = {
        "optimization_history.html": plot_optimization_history(study),
        "param_importances.html": plot_param_importances(study),
        "parallel_coordinates.html": plot_parallel_coordinate(study),
    }

    for filename, fig in plots.items():
        fig.write_html(f"{output_path}/{filename}")
        logging.info(f"Saved {filename}")
        time.sleep(0.25)


def cleanup_study_file(study_name: str, storage_dir=OPTUNA_STUDIES_FOLDER):
    """
    Delete the Optuna study file with the given name from the default storage directory.

    Args:
        study_name (str): The name of the Optuna study to delete.
        storage_dir (str, optional): The directory where Optuna study files are stored. Defaults to "optuna_studies".
    """
    path = os.path.join(storage_dir, f"{study_name}.db")

    if os.path.exists(path):
        os.remove(path)
        logging.info(f"Deleted study file: {path}")
    else:
        logging.info(f"Study file not found: {path}")


def db_writer_process(queue: mp.Queue, done_flag, ack_dict: dict):
    sql = BacktestSQLHelper()
    cache_sql = IndicatorCacheSQLHelper()

    while True:
        while not queue.empty():
            queue_dict: dict = queue.get()

            # Determine why queue item was inserted
            purpose = queue_dict.get("purpose")

            try:
                if purpose == "strategy_config":
                    strategy_config_id = sql.insert_strategy_configuration(
                        queue_dict.get("strategy_name"),
                        queue_dict.get("strategy_description"),
                        queue_dict.get("strategy_parameters"),
                    )
                    if ack_dict is not None and queue_dict.get("ack_id"):
                        ack_dict[queue_dict["ack_id"]] = True

                elif purpose == "indicator_cache":
                    cache_items: list[dict] = queue_dict.get("cache_items")
                    cache_sql.insert_cache_items(cache_items)

                # Write trial to DB (backtest & score || score)
                elif purpose == "trial":
                    # Unpack queue item
                    pair_id = queue_dict.get("pair_id")
                    strategy: strategies.NNFXStrategy = queue_dict.get("strategy")
                    timeframe = queue_dict.get("timeframe")
                    metrics_df = queue_dict.get("metrics_df")
                    study_name = queue_dict.get("study_name")
                    trial_id = queue_dict.get("trial_id")
                    score = queue_dict.get("score")
                    exploration_space = queue_dict.get("exploration_space")
                    trial_start = queue_dict.get("trial_start")

                    # Select strategy configuration
                    strategy_config_id = sql.select_strategy_configuration(
                        strategy.NAME,
                        strategy.DESCRIPTION,
                        strategy.PARAMETER_SETTINGS,
                    )

                    # Check if run already exists
                    start_date, end_date = (
                        metrics_df["Data_Start_Date"].iloc[0],
                        metrics_df["Data_End_Date"].iloc[0],
                    )
                    run_id = sql.select_backtest_run_by_config(
                        pair_id, strategy_config_id, timeframe, start_date, end_date
                    )

                    # Insert run if it doesn't exist
                    if run_id is None and metrics_df is not None:
                        run_id = sql.insert_backtest_run(
                            pair_id, strategy_config_id, timeframe, metrics_df
                        )
                        # logging.info(f"Inserted new run {run_id}")

                    # Insert composite score
                    if run_id is not None and score is not None:
                        sql.insert_composite_score(
                            run_id,
                            study_name,
                            None,
                            trial_id,
                            score,
                            exploration_space,
                            trial_start,
                        )
                        # logging.info("Saved composite score.")

                    if run_id is None and score is not None:
                        raise ValueError("Run ID is None but score is not None.")

                # Write study metadata and backfill composite scores
                elif purpose == "study":
                    # Write study metadata
                    meta_df = queue_dict.get("meta_df")
                    sql.insert_study_metadata(meta_df)
                    study_id = sql.select_study_id(meta_df["Study_Name"][0])

                    # Backfill composite score table to link to study metadata
                    sql.backfill_composite_scores(study_id)

                # For phase 3 runs
                elif purpose == "backtest_run":
                    # Unpack queue item
                    pair_id = queue_dict.get("pair_id")
                    strategy_config_id = queue_dict.get("strategy_config_id")
                    timeframe = queue_dict.get("timeframe")
                    metrics_df = queue_dict.get("metrics_df")
                    study_name = queue_dict.get("study_name")
                    score = queue_dict.get("score")
                    exploration_space = queue_dict.get("exploration_space")

                    # Check if run already exists
                    start_date, end_date = (
                        metrics_df["Data_Start_Date"].iloc[0],
                        metrics_df["Data_End_Date"].iloc[0],
                    )
                    run_id = sql.select_backtest_run_by_config(
                        pair_id, strategy_config_id, timeframe, start_date, end_date
                    )

                    # Insert run if it doesn't exist
                    if run_id is None and metrics_df is not None:
                        run_id = sql.insert_backtest_run(
                            pair_id, strategy_config_id, timeframe, metrics_df
                        )
                        # logging.info(f"Inserted new run {run_id}")

                    # Insert composite score
                    if run_id is not None and score is not None:
                        sql.insert_composite_score(
                            run_id,
                            study_name,
                            -1,
                            -1,
                            score,
                            exploration_space,
                            None,
                        )
                        # logging.info("Saved composite score.")

                    if run_id is None and score is not None:
                        raise ValueError("Run ID is None but score is not None.")

            except Exception as e:
                log_error(f"DB write error: {e} {traceback.format_exc()}")
                winsound.Beep(300, 500)

        if done_flag.value is True:
            # Close DB connection
            sql.close_connection()
            cache_sql.close_connection()
            del sql
            del cache_sql
            break


def build_indicator_config_for_trial(
    trial: optuna.Trial,
    role: str,  # ATR, Baseline, etc
    candidate_configs: list[
        IndicatorConfig
    ],  # List of available indicators for this role
    use_default_params: bool,
) -> IndicatorConfig:
    # Pool of available indicators for this role
    name_pool = [c["name"] for c in candidate_configs]
    selected_name = trial.suggest_categorical(role, name_pool)

    config = next(c for c in candidate_configs if c["name"] == selected_name)

    if use_default_params:
        sampled_params = config[
            "parameters"
        ]  # just reuse the default config's parameter values
    else:
        sampled_params = {}
        for param_name, space in config.get("parameter_space", {}).items():
            # Unique key for Optuna tracking (i.e. C1.Laguerre.timeperiod)
            key = f"{role}.{selected_name}.{param_name}"
            if not isinstance(space, list):
                raise ValueError(
                    f"Expected list for parameter space: {role}.{param_name}"
                )

            sampled_params[param_name] = trial.suggest_categorical(key, space)

    return {
        "name": selected_name,
        "function": config["function"],
        "signal_function": config["signal_function"],
        "raw_function": config["raw_function"],
        "description": config.get("description", ""),
        "parameters": sampled_params,
    }


def objective(
    trial: optuna.Trial,
    forex_pair: str,
    timeframe: str,
    db_queue: mp.Queue,
    ack_dict: dict,
    allowed_indicators: dict,
    indicator_config_spaces: dict[str, list[IndicatorConfig]],
) -> float:
    """
    Optuna objective function to maximize composite score of an NNFX strategy.
    Now using safe categorical values (strings) + dictionary lookup.
    """
    # Check if we're exploring the full indicator space (Phase 1)
    use_default_params = all(
        len(v) > 1 and v[0] == "all" for v in allowed_indicators.values()
    )

    atr = build_indicator_config_for_trial(
        trial, "ATR", indicator_config_spaces["ATR"], use_default_params
    )
    baseline = build_indicator_config_for_trial(
        trial, "Baseline", indicator_config_spaces["Baseline"], use_default_params
    )
    c1 = build_indicator_config_for_trial(
        trial, "C1", indicator_config_spaces["C1"], use_default_params
    )
    c2 = build_indicator_config_for_trial(
        trial, "C2", indicator_config_spaces["C2"], use_default_params
    )
    volume = build_indicator_config_for_trial(
        trial, "Volume", indicator_config_spaces["Volume"], use_default_params
    )
    exit = build_indicator_config_for_trial(
        trial, "Exit", indicator_config_spaces["Exit"], use_default_params
    )

    # Prevent invalid C1/C2 combo
    if c1["name"] == c2["name"]:
        raise optuna.exceptions.TrialPruned()

    # Initialize strategy and backtester
    backtest_sqlhelper = BacktestSQLHelper(read_only=True)
    strategy = strategies.NNFXStrategy(
        atr, baseline, c1, c2, volume, exit, forex_pair, timeframe
    )

    # Get strategy config ID if it already exists
    strategy_config_id = backtest_sqlhelper.select_strategy_configuration(
        strategy.NAME,
        strategy.DESCRIPTION,
        strategy.PARAMETER_SETTINGS,
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

        # Wait for the job to be acknowledged
        wait_for_ack(ack_dict, ack_id)  # Blocks until confirmed
        strategy_config_id = backtest_sqlhelper.select_strategy_configuration(
            strategy.NAME,
            strategy.DESCRIPTION,
            strategy.PARAMETER_SETTINGS,
        )

    # Check config for how to slice historical data based on timeframe
    timeframe_range = TIMEFRAME_DATE_RANGES[timeframe]
    from_date, to_date = timeframe_range["from_date"], timeframe_range["to_date"]

    # Load historical OHLCV + indicator data
    data_sqlhelper = HistoricalDataSQLHelper(
        f"{DATA_FOLDER}/{forex_pair.replace('/', '')}.db", read_only=True
    )
    data: pd.DataFrame = data_sqlhelper.get_historical_data(
        table=timeframe, from_date=from_date, to_date=to_date
    )

    # Prepare data for strategy
    strategy.prepare_data(data)

    # Get cache jobs to insert into DB
    cache_items: list[dict] = strategy.get_cache_jobs()

    # Insert cache jobs into DB
    if len(cache_items) > 0:
        db_queue.put({"purpose": "indicator_cache", "cache_items": cache_items})

    # Initialize backtester
    backtester = Backtester(
        strategy,
        forex_pair,
        timeframe,
        data,
        initial_balance=10_000,
    )

    pair_id = None
    run_metrics_df = None
    score = None
    manually_pruned = False
    ran_new_backtest = False
    calculated_score = False

    try:
        # Check if this configuration has already been run
        pair_id = backtest_sqlhelper.select_forex_pair_id(forex_pair)
        run_id = backtest_sqlhelper.select_backtest_run_by_config(
            pair_id, strategy_config_id, timeframe, from_date, to_date
        )

        # If run already exists, check if we've calculated composite score
        if run_id is not None:
            score = backtest_sqlhelper.select_composite_score(run_id)
            # Prune trials with existing score
            if score is not None:
                # logging.info("Pruned trial due to existing run.")
                manually_pruned = True
                raise optuna.exceptions.TrialPruned()
            # Get existing metrics to calculate composite score
            elif score is None:
                run_metrics_df = backtest_sqlhelper.select_backtest_run_metrics_by_id(
                    run_id
                )
                backtester.set_metrics_df(run_metrics_df)
                # logging.info("Retrieved existing run metrics for score calculation.")

        # Run does not exist
        else:
            # Run backtest
            # logging.info(f"[{forex_pair}-{timeframe}] Running backtest...")
            try:
                backtester.run_backtest()
            except Exception as e:
                log_error(
                    f"[{forex_pair}-{timeframe}] PRUNING trial due to backtest failing with error: {e}"
                )
                manually_pruned = True
                raise optuna.exceptions.TrialPruned()

            # Retrieve metrics
            run_metrics_df = backtester.get_metrics_df()

            ran_new_backtest = True

        backtest_sqlhelper.close_connection()
        backtest_sqlhelper = None

        # Prune trials with too few trades to minimize exploring inactive configurations
        min_trades_per_day = MIN_TRADES_PER_DAY[timeframe]
        prune_threshold = round(min_trades_per_day * PRUNE_THRESHOLD_FACTOR, 2)
        trades_per_day = backtester.get_metric("Trades_Per_Day")
        if trades_per_day < prune_threshold:
            # logging.info(
            #     f"Pruned trial due to low activity: {trades_per_day:.2f} trades/day < required {prune_threshold} trades/day for {timeframe} timeframe"
            # )
            manually_pruned = True
            raise optuna.exceptions.TrialPruned()

        # Calculate composite score
        score = backtester.calculate_composite_score()
        calculated_score = True

        # Replace inf or NaN with None
        if score is None or score != score or score in (float("inf"), float("-inf")):
            logging.warning("Pruned trial due to invalid score.")
            manually_pruned = True
            score = None
            raise optuna.exceptions.TrialPruned()

        # Optimize for higher score
        return score
    except Exception as e:
        # If manually pruned, let error raise normally so Optuna handles it and trials continue
        if manually_pruned:
            raise e

        full_traceback = traceback.format_exc()
        log_error(f"Fatal error on {forex_pair} {timeframe}: {full_traceback}")
        winsound.Beep(200, 1000)
        sys.exit(1)
    finally:
        if ran_new_backtest or (not ran_new_backtest and calculated_score):
            trial_id = trial.number
            trial_start = (
                trial.datetime_start.isoformat() if trial.datetime_start else None
            )
            study_name = trial.study.study_name

            # Find word after 'space-' and before next underscore
            exploration_space = study_name.split("space-")[1].split("_")[0]

            db_queue.put(
                {
                    "pair_id": pair_id,
                    "strategy": strategy,
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


def get_study_meta(
    study: optuna.study.Study, pair: str, timeframe: str, exploration_space: str
) -> dict:
    """Get study metadata

    Args:
        study (optuna.study.Study): The study to get metadata from
        n_trials (int): The number of trials in the study

    Returns:
        dict: The study metadata
    """
    first_trial = study.trials[0]
    last_completed_trial = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ][-1]
    start_time = first_trial.datetime_start.timestamp()
    duration = last_completed_trial.datetime_complete.timestamp() - start_time
    best_trial = study.best_trial
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]

    score_values = [
        t.value for t in completed_trials if isinstance(t.value, (float, int))
    ]

    meta = {
        "Study_Name": study.study_name,
        "Study": study,
        "Pair": pair,
        "Timeframe": timeframe,
        "Exploration_Space": exploration_space,
        "Best_Score": best_trial.value,
        "Best_Trial": best_trial.number,
        "Time_To_Best": (
            best_trial.datetime_complete.timestamp() - start_time
            if best_trial.datetime_complete
            else None
        ),
        "Total_Time_Sec": round(duration, 2),
        "N_Trials": len(study.trials),
        "N_Completed": len(completed_trials),
        "N_Pruned": len(pruned_trials),
        "Avg_Score": round(mean(score_values), 4) if score_values else None,
        "Std_Score": round(stdev(score_values), 4) if len(score_values) > 1 else None,
        "Stop_Reason": study.user_attrs.get("stop_reason", "max_trials_reached"),
    }

    return meta


def run_study_wrapper(args) -> dict:
    (
        pair,
        timeframe,
        n_trials,
        exploration_space,
        phase_name,
        seed,
        allowed_indicators,
        indicator_config_spaces,
        db_queue,
        ack_dict,
    ) = args
    """Wrapper for multiprocessing

    Args:
        args (Tuple): A tuple of arguments (pair, timeframe, n_trials, exploration_space, db_queue, phase_name, seed)
    Returns:
        str: The study name
    """
    logging.info(f"[START STUDY] {pair} @ {timeframe}")

    def wrapped_objective(trial):
        return objective(
            trial,
            forex_pair=pair,
            timeframe=timeframe,
            db_queue=db_queue,
            ack_dict=ack_dict,
            allowed_indicators=allowed_indicators,
            indicator_config_spaces=indicator_config_spaces,
        )

    timeframe_range = TIMEFRAME_DATE_RANGES[timeframe]
    from_date, to_date = timeframe_range["from_date"], timeframe_range["to_date"]

    # callback_n = IMPROVEMENT_CUTOFF_BY_TIMEFRAME[timeframe]

    study_name = (
        f"{phase_name}_{pair.replace('/', '')}_{timeframe}_NNFX_"
        f"{from_date}to{to_date}_space-{exploration_space}_seed-{seed}"
    )

    # Check if study already has already completed/stopped (if exists)
    study_metadata: pd.Series = BacktestSQLHelper(read_only=True).select_study_metadata(
        study_name
    )
    stop_reason = study_metadata["Stop_Reason"] if study_metadata is not None else None
    if stop_reason:
        logging.info(f"Skipping study creation: [{study_name}] - {stop_reason}")
        return None

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=TPESampler(
            seed=seed,
            n_startup_trials=max(10, int(n_trials * N_STARTUP_TRIALS_PERCENTAGE)),
        ),  # n_startup_trials indicates the number of initial random trials
        storage=f"sqlite:///{OPTUNA_STUDIES_FOLDER}/{study_name}.db",
        load_if_exists=True,
    )

    # Maintain upper bound on number of trials after study resumption
    completed_trials = [
        t
        for t in study.trials
        if t.state
        not in [
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.WAITING,
        ]
    ]
    remaining_trials = max(n_trials - len(completed_trials), 0)

    # Run study
    study.optimize(
        wrapped_objective,
        n_trials=remaining_trials,
        n_jobs=1,
        # callbacks=[early_stop_callback(callback_n)],
    )

    # Get study metadata
    meta = get_study_meta(study, pair, timeframe, exploration_space)

    logging.info(
        f"[FINISHED STUDY] {pair} @ {timeframe} | Best score: {meta['Best_Score']:.3f} at trial {meta['Best_Trial']}"
    )
    return meta


def run_all_studies(
    study_args_list: list[tuple],
    max_parallel_studies: int,
):
    """Run Optuna studies on all pair/timeframe combos

    Args:
        n_trials_per_combo (int, optional): The number of trials per pair/timeframe combo. Defaults to 100.
        max_parallel_studies (int, optional): The maximum number of parallel studies. Defaults to 1.
        exploration_space (str, optional): The exploration space to use.
            Can be "default" to test default indicator settings, "parameterized" to test parameterized indicator settings.
    """
    manager = mp.Manager()
    db_queue = manager.Queue()
    ack_dict = manager.dict()
    done_flag = mp.Value("b", False)

    # Start DB writer process
    writer_proc = mp.Process(
        target=db_writer_process, args=(db_queue, done_flag, ack_dict)
    )
    writer_proc.start()

    meta_results = []
    try:
        # Prepare study jobs
        # Append DB-related args to each study job
        study_args = [args + (db_queue, ack_dict) for args in study_args_list]

        with ProcessPoolExecutor(max_workers=max_parallel_studies) as executor:
            futures = {
                executor.submit(run_study_wrapper, args): args for args in study_args
            }

            # Yield results in order of completion rather than insertion
            for future in as_completed(futures):
                args = futures[future]
                try:
                    meta = future.result()
                    if meta is None:
                        continue

                    meta_results.append(meta)
                    meta_df = pd.DataFrame(
                        [{k: v for k, v in meta.items() if k != "Study"}]
                    )
                    db_queue.put({"purpose": "study", "meta_df": meta_df})

                except Exception as e:
                    error_msg = f"[ERROR] Study failed for args: {args} | Error: {e} | Traceback: {traceback.format_exc()}"
                    logging.error(error_msg)
                    log_error(error_msg)
    finally:
        done_flag.value = True
        writer_proc.join()

    logging.info(f"All ({len(meta_results)}) studies completed, cleaning up...")
    for meta in meta_results:
        # Delete study object to release references to study DB
        del meta["Study"]
        gc.collect()

    logging.info("Done.")

    winsound.Beep(200, 1000)


def build_study_args_phase1(
    pairs: list[str],
    timeframes: list[str],
    seeds: list[int],
    trials_by_timeframe: dict[str, int],
    exploration_space: str = "default",
    phase_name: str = "phase1",
) -> list[tuple]:
    # Use full search space of indicator names
    allowed_indicators = {
        "ATR": ["all"],
        "Baseline": ["all"],  # LSMA
        "C1": ["all"],
        "C2": ["all"],
        "Volume": ["all"],
        "Exit": ["all"],
    }

    # Full indicator config sets for name selection
    indicator_space = {
        "ATR": atr_indicators,
        "Baseline": baseline_indicators,
        "C1": trend_indicators + momentum_indicators,
        "C2": trend_indicators + momentum_indicators,
        "Volume": volume_indicators,
        "Exit": trend_indicators + momentum_indicators,
    }

    args = []
    for seed in seeds:
        for pair in pairs:
            for timeframe in timeframes:
                if timeframe not in trials_by_timeframe:
                    continue
                trials = trials_by_timeframe[timeframe]
                args.append(
                    (
                        pair,
                        timeframe,
                        trials,
                        exploration_space,
                        phase_name,
                        seed,
                        allowed_indicators,
                        indicator_space,
                    )
                )
    return args


def build_study_args_phase2(
    pairs: list[str],
    timeframes: list[str],
    seeds: list[int],
    trials_by_timeframe: dict[str, int],
    top_percent: float = PHASE2_TOP_PERCENT,
    exploration_space: str = f"top_{PHASE2_TOP_PERCENT}percent_parameterized",
    phase_name: str = "phase2",
) -> list[tuple]:
    # Get top N% performing strategies
    db = BacktestSQLHelper(read_only=True)
    top_strategies = db.select_top_percent_strategies("default", top_percent)

    # Group top indicator names per role
    role_to_names: dict[str, set[str]] = {
        "ATR": set(),
        "Baseline": set(),
        "C1": set(),
        "C2": set(),
        "Volume": set(),
        "Exit": set(),
    }

    for strat in top_strategies:
        params = strat["Parameters"]
        for role in role_to_names:
            param = params.get(role.lower(), {})
            name = param.split("_")[0]
            if name:
                role_to_names[role].add(name)

    # Now filter your full indicator pools to only include the ones in the top sets
    def filter_by_name(configs: list[dict], names: set[str]):
        return [cfg for cfg in configs if cfg["name"] in names]

    indicator_config_spaces = {
        "ATR": filter_by_name(atr_indicators, role_to_names["ATR"]),
        "Baseline": filter_by_name(baseline_indicators, role_to_names["Baseline"]),
        "C1": filter_by_name(
            trend_indicators + momentum_indicators, role_to_names["C1"]
        ),
        "C2": filter_by_name(
            trend_indicators + momentum_indicators, role_to_names["C2"]
        ),
        "Volume": filter_by_name(volume_indicators, role_to_names["Volume"]),
        "Exit": filter_by_name(
            trend_indicators + momentum_indicators, role_to_names["Exit"]
        ),
    }

    allowed_indicators = {role: list(names) for role, names in role_to_names.items()}

    # Create study args per pair/timeframe/seed
    args = []
    for seed in seeds:
        for pair in pairs:
            for timeframe in timeframes:
                if timeframe not in trials_by_timeframe:
                    continue
                trials = trials_by_timeframe[timeframe]
                args.append(
                    (
                        pair,
                        timeframe,
                        trials,
                        exploration_space,
                        phase_name,
                        seed,
                        allowed_indicators,
                        indicator_config_spaces,
                    )
                )

    return args


def build_study_args_phase3(
    pairs: list[str],
    timeframes: list[str],
    top_n: int = 250,
) -> list[tuple]:
    import ast

    def parse_indicator_field(value: str):
        match = re.match(r"^(.*?)_\{(.*)\}$", value)
        if not match:
            return value, {}
        name = match.group(1)
        param_str = "{" + match.group(2) + "}"
        try:
            params = ast.literal_eval(param_str)
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
    top_strategies = db.select_top_n_strategies_across_studies(top_n=top_n)
    existing_combos = db.get_existing_scored_combinations()

    args = []

    for strategy_row in top_strategies:
        raw_params = strategy_row["Parameters"]
        role_configs = {}

        for raw_key, full_str in raw_params.items():
            role = key_map.get(raw_key)
            if not role:
                continue
            name, params = parse_indicator_field(full_str)

            if role == "ATR":
                pool = atr_indicators
            elif role == "Baseline":
                pool = baseline_indicators
            elif role in ["C1", "C2", "Exit"]:
                pool = trend_indicators + momentum_indicators
            elif role == "Volume":
                pool = volume_indicators
            else:
                continue

            matched = next((cfg for cfg in pool if cfg["name"] == name), None)
            if not matched:
                logging.warning(f"Indicator '{name}' not found for role {role}")
                break

            # Check if matched parameters are same as the ones from the database
            if matched["parameters"] != params:
                x = 1

            role_configs[role] = {**matched, "parameters": params}

        if len(role_configs) != 6:
            continue  # Skip incomplete strategy

        for pair in pairs:
            for timeframe in timeframes:
                strategy_obj = strategies.NNFXStrategy(
                    atr=role_configs["ATR"],
                    baseline=role_configs["Baseline"],
                    c1=role_configs["C1"],
                    c2=role_configs["C2"],
                    volume=role_configs["Volume"],
                    exit_indicator=role_configs["Exit"],
                    forex_pair=pair,
                    timeframe=timeframe,
                )

                if (
                    strategy_row["StrategyConfig_ID"],
                    pair,
                    timeframe,
                ) in existing_combos:
                    continue

                args.append(
                    (
                        pair,
                        timeframe,
                        strategy_obj,
                    )
                )

    return args


def run_phase3(phase: dict):
    run_args = build_study_args_phase3(
        pairs=MAJOR_FOREX_PAIRS,
        timeframes=TIMEFRAMES,
        top_n=phase["top_n"],
    )

    logging.info(
        f"Running {len(run_args)} fixed strategy evaluations across all pair/timeframe combos..."
    )

    manager = mp.Manager()
    db_queue = manager.Queue()
    ack_dict = manager.dict()
    done_flag = mp.Value("b", False)

    writer_proc = mp.Process(
        target=db_writer_process, args=(db_queue, done_flag, ack_dict)
    )
    writer_proc.start()

    completed = 0
    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        futures = []
        for pair, timeframe, strategy in run_args:
            futures.append(
                executor.submit(
                    run_fixed_strategy_evaluation,
                    strategy,
                    pair,
                    timeframe,
                    phase["name"],
                    phase["exploration_space"],
                    db_queue,
                )
            )

        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                logging.info(f"Progress: {completed}/{len(run_args)} completed")
            except Exception as e:
                logging.error(f"[ERROR] Phase 3 job failed: {e}")

    done_flag.value = True
    writer_proc.join()
    logging.info("Phase 3 complete.")

    return


def correct_and_rerun_lsma_strategies_fixed_eval(json_path: str, db_queue: mp.Queue):
    """
    Loads best_strategies.json, finds LSMA strategies with shift ≠ 0,
    deletes them from the DB, sets shift = 0, and re-evaluates each using
    run_fixed_strategy_evaluation.
    """
    with open(json_path, "r") as f:
        strategy_list = json.load(f)

    sql = BacktestSQLHelper()
    updated_strategies = []

    for strat in strategy_list:
        indicators = strat["indicators"]
        updated = False

        for key, ind in indicators.items():
            if ind["name"] == "LSMA":
                print(
                    f"⚠️ LSMA with shift={ind['parameters']['shift']} found in strategy ID {strat['strategy_id']}"
                )

                # Reset shift to 0
                strat["indicators"][key]["parameters"]["shift"] = 0
                updated = True

        if updated:
            strategy_id = strat["strategy_id"]

            # Delete old configuration (safe match based on original ID or exact parameters)
            # Step 1: Find all BacktestRun IDs tied to this strategy
            sql.safe_execute(
                sql.cursor,
                "SELECT id FROM BacktestRuns WHERE Strategy_ID = ?",
                (strategy_id,),
            )
            backtest_ids = [row[0] for row in sql.cursor.fetchall()]

            # Step 2: Delete CompositeScores tied to those backtests
            if backtest_ids:
                placeholders = ",".join("?" for _ in backtest_ids)
                sql.safe_execute(
                    sql.cursor,
                    f"DELETE FROM CompositeScores WHERE Backtest_ID IN ({placeholders})",
                    backtest_ids,
                )

            # Step 3: Delete BacktestRuns
            sql.safe_execute(
                sql.cursor,
                "DELETE FROM BacktestRuns WHERE Strategy_ID = ?",
                (strategy_id,),
            )

            # Step 4: Delete StrategyConfiguration
            sql.safe_execute(
                sql.cursor,
                "DELETE FROM StrategyConfigurations WHERE id = ?",
                (strategy_id,),
            )

            print(f"🗑 Fully deleted strategy {strategy_id} and dependencies.")

            # Reload and evaluate
            print(f"🔄 Re-evaluating strategy {strat['strategy_id']} with LSMA shift=0")
            strategy_obj = utilities.load_strategy_from_dict(strat)

            sql.insert_strategy_configuration(
                strategy_obj.NAME,
                strategy_obj.DESCRIPTION,
                strategy_obj.PARAMETER_SETTINGS,
            )

            run_fixed_strategy_evaluation(
                strategy=strategy_obj,
                pair=strat["pair"],
                timeframe=strat["timeframe"],
                phase_name="Phase4Fix",
                exploration_space="manual_fix",
                db_queue=db_queue,
            )

            updated_strategies.append(strat)

    sql.conn.commit()
    sql.close_connection()
    print(f"✅ Reran {len(updated_strategies)} corrected LSMA strategies with shift=0.")


if __name__ == "__main__":
    # Run Optuna studies on all pair/timeframe combos
    N_PROCESSES = mp.cpu_count() - 4

    # sql = BacktestSQLHelper()
    # sql.delete_strategies_lsma_nonzero_shift()

    # final_best_results = BacktestSQLHelper(
    #     read_only=True
    # ).get_best_strategy_per_pair_with_metrics()

    # # Save to JSON file
    # with open("best_strategies.json", "w") as f:
    #     json.dump(final_best_results, f, indent=2)

    # # Initialize multiprocessing queue for DB writes
    # manager = mp.Manager()
    # db_queue = manager.Queue()
    # ack_dict = manager.dict()
    # done_flag = mp.Value("b", False)

    # # Path to your best strategies JSON file
    # json_path = "best_strategies.json"

    # # Run the LSMA shift=-1 fix + re-evaluation
    # correct_and_rerun_lsma_strategies_fixed_eval(json_path, db_queue)

    # # Now start the writer after queue is fully populated
    # done_flag.value = False  # Just to be explicit
    # writer_proc = mp.Process(
    #     target=db_writer_process, args=(db_queue, done_flag, ack_dict)
    # )
    # writer_proc.start()
    # writer_proc.join()
    # print("✅ All strategy results written safely.")

    PHASES = [
        {
            "name": "phase1",
            "exploration_space": "default",
            "seeds": [42, 1337, 314, 777, 444],
            "trials_by_timeframe": {
                "1_day": 500,
                "4_hour": 450,
                "2_hour": 400,
            },
        },
        {
            "name": "phase2",
            "exploration_space": f"top_{PHASE2_TOP_PERCENT}percent_parameterized",
            "seeds": [42, 1337],
            "trials_by_timeframe": {
                "1_day": 500,
                "4_hour": 450,
                "2_hour": 400,
            },
            "top_percent": PHASE2_TOP_PERCENT,
        },
        {
            "name": "phase3",
            "exploration_space": "generalization_test",
            "seeds": [42],  # Arbitrary seed; params are fixed
            "trials_by_timeframe": {
                "1_day": 1,
                "4_hour": 1,
                "2_hour": 1,
            },
            "top_n": 250,
        },
    ]

    logging.info(f"Running phases {[phase['name'] for phase in PHASES]}...")

    # Get best pair/timeframe combos bast on previous phase runs
    sql = BacktestSQLHelper(read_only=True)
    filtered_pairs_timeframes = sql.get_top_pair_timeframes_by_best_score(
        min_best_score=13
    )

    # Use in kwargs for study setup
    filtered_pairs = sorted(set(pair for pair, tf in filtered_pairs_timeframes))
    filtered_timeframes = sorted(set(tf for pair, tf in filtered_pairs_timeframes))

    for phase in PHASES:
        logging.info(f"Running phase {phase['name']}...")
        name = phase["name"]
        kwargs = {
            "phase_name": name,
            "pairs": filtered_pairs,
            "timeframes": filtered_timeframes,
        }

        if name == "phase1":
            # continue
            kwargs["exploration_space"] = phase["exploration_space"]
            kwargs["trials_by_timeframe"] = phase["trials_by_timeframe"]
            kwargs["seeds"] = phase["seeds"]
            study_args = build_study_args_phase1(**kwargs)

            logging.info(
                f"Running {len(study_args)} total jobs across all pair/timeframe/seed combos..."
            )

            run_all_studies(study_args, max_parallel_studies=N_PROCESSES)

        elif name == "phase2":
            kwargs["exploration_space"] = phase["exploration_space"]
            kwargs["trials_by_timeframe"] = phase["trials_by_timeframe"]
            kwargs["seeds"] = phase["seeds"]
            kwargs["top_percent"] = phase["top_percent"]
            study_args = build_study_args_phase2(**kwargs)

            logging.info(
                f"Running {len(study_args)} total jobs across all pair/timeframe/seed combos..."
            )

            run_all_studies(study_args, max_parallel_studies=N_PROCESSES)

        elif name == "phase3":
            # continue
            run_phase3(phase)

        else:
            raise ValueError(f"Unknown phase name: {name}")

    final_best_results = BacktestSQLHelper(
        read_only=True
    ).get_best_strategy_per_pair_with_metrics()

    # Save to JSON file
    with open("best_strategies.json", "w") as f:
        json.dump(final_best_results, f, indent=2)
