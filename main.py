import datetime
import gc
import logging
import multiprocessing as mp
import os
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

from scripts import strategies
from scripts.backtester import Backtester
from scripts.config import (
    DATA_FOLDER,
    MAJOR_FOREX_PAIRS,
    MIN_TRADES_PER_DAY,
    N_STARTUP_TRIALS_PERCENTAGE,
    OPTUNA_STUDIES_FOLDER,
    PHASES,
    PRUNE_THRESHOLD_FACTOR,
    TEMP_CACHE_FOLDER,
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


def get_indicator_by_name(name: str, pool: list[dict]) -> IndicatorConfig:
    """Find an indicator config by name from a given pool."""
    for ind in pool:
        if ind["name"] == name:
            return ind
    raise ValueError(f"Indicator with name '{name}' not found.")


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
                    strategy_config_id = sql.upsert_strategy_configuration(
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

            except Exception as e:
                logging.error(f"DB write error: {e} {traceback.format_exc()}")
                winsound.Beep(300, 500)
                # Close DB connection
                sql.close_connection()
                cache_sql.close_connection()
                del sql
                del cache_sql

        if done_flag.value:
            # Close DB connection
            sql.close_connection()
            cache_sql.close_connection()
            del sql
            del cache_sql
            break


def objective(
    trial: optuna.Trial,
    forex_pair: str,
    timeframe: str,
    db_queue: mp.Queue,
    ack_dict: dict,
) -> float:
    """
    Optuna objective function to maximize composite score of an NNFX strategy.
    Now using safe categorical values (strings) + dictionary lookup.
    """
    # Choose by name to allow Optuna to persist trials cleanly
    atr_name = trial.suggest_categorical("ATR", [i["name"] for i in atr_indicators])
    baseline_name = trial.suggest_categorical(
        "Baseline", [i["name"] for i in baseline_indicators]
    )
    c1_name = trial.suggest_categorical(
        "C1", [i["name"] for i in trend_indicators + momentum_indicators]
    )
    c2_name = trial.suggest_categorical(
        "C2", [i["name"] for i in trend_indicators + momentum_indicators]
    )
    volume_name = trial.suggest_categorical(
        "Volume", [i["name"] for i in volume_indicators]
    )
    exit_name = trial.suggest_categorical(
        "Exit", [i["name"] for i in trend_indicators + momentum_indicators]
    )

    # Prevent invalid C1/C2 combo
    if c1_name == c2_name:
        raise optuna.exceptions.TrialPruned()

    # Retrieve full config dicts by name
    atr = get_indicator_by_name(atr_name, atr_indicators)
    baseline = get_indicator_by_name(baseline_name, baseline_indicators)
    c1 = get_indicator_by_name(c1_name, trend_indicators + momentum_indicators)
    c2 = get_indicator_by_name(c2_name, trend_indicators + momentum_indicators)
    volume = get_indicator_by_name(volume_name, volume_indicators)
    exit = get_indicator_by_name(exit_name, trend_indicators + momentum_indicators)

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
        pair_id = backtest_sqlhelper.get_forex_pair_id(forex_pair)
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
            backtester.run_backtest()

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
        logging.error(f"Fatal error on {forex_pair} {timeframe}: {full_traceback}")
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
        db_queue,
        ack_dict,
    ) = args
    """Wrapper for multiprocessing

    Args:
        args (Tuple): A tuple of arguments (pair, timeframe, n_trials, exploration_space, db_queue, phase_name, seed)
    Returns:
        str: The study name
    """
    (
        pair,
        timeframe,
        n_trials,
        exploration_space,
        phase_name,
        seed,
        db_queue,
        ack_dict,
    ) = args

    logging.info(f"[START STUDY] {pair} @ {timeframe}")

    def wrapped_objective(trial):
        return objective(trial, pair, timeframe, db_queue, ack_dict)

    timeframe_range = TIMEFRAME_DATE_RANGES[timeframe]
    from_date, to_date = timeframe_range["from_date"], timeframe_range["to_date"]

    # callback_n = IMPROVEMENT_CUTOFF_BY_TIMEFRAME[timeframe]

    study_name = (
        f"{phase_name}_{pair.replace('/', '')}_{timeframe}_NNFX_"
        f"{from_date}to{to_date}_space-{exploration_space}_seed-{seed}"
    )

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

    # Check if study already has already completed/stopped (if exists)
    study_metadata: pd.Series = BacktestSQLHelper(read_only=True).select_study_metadata(
        study_name
    )
    stop_reason = study_metadata["Stop_Reason"] if study_metadata is not None else None
    if stop_reason:
        logging.info(f"[{study.study_name}] Already stopped: {stop_reason}")
        return None

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
        study_args = [
            (pair, tf, trials, space, phase_name, seed, db_queue, ack_dict)
            for (pair, tf, trials, space, phase_name, seed) in study_args_list
        ]

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


if __name__ == "__main__":
    # Run Optuna studies on all pair/timeframe combos
    N_PROCESSES = mp.cpu_count() - 4

    logging.info(f"Running phases {[phase['name'] for phase in PHASES]}...")

    all_study_jobs = []

    for phase in PHASES:
        if phase["name"] == "phase2":
            continue

        for seed in phase["seeds"]:
            for pair in MAJOR_FOREX_PAIRS:
                for timeframe in TIMEFRAMES:
                    if timeframe not in phase["trials_by_timeframe"]:
                        continue

                    all_study_jobs.append(
                        (
                            pair,
                            timeframe,
                            phase["trials_by_timeframe"][timeframe],
                            phase["exploration_space"],
                            phase["name"],
                            seed,
                        )
                    )

    logging.info(
        f"Running {len(all_study_jobs)} total jobs across all pair/timeframe/seed combos..."
    )

    run_all_studies(all_study_jobs, max_parallel_studies=N_PROCESSES)
