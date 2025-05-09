import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
import winsound
from concurrent.futures import ProcessPoolExecutor

import optuna
from optuna.samplers import TPESampler
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)

from scripts import config, strategies
from scripts.backtester import Backtester
from scripts.config import MAJOR_FOREX_PAIRS, TIMEFRAMES
from scripts.indicators import (
    atr_indicators,
    baseline_indicators,
    momentum_indicators,
    trend_indicators,
    volume_indicators,
)
from scripts.sql import BacktestSQLHelper

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)

# db_queue = mp.Queue()
# done_flag = mp.Value("b", False)


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


def get_indicator_by_name(name: str, pool: list[dict]) -> dict:
    """Find an indicator config by name from a given pool."""
    for ind in pool:
        if ind["name"] == name:
            return ind
    raise ValueError(f"Indicator with name '{name}' not found.")


def db_writer_process(queue: mp.Queue, done_flag):
    """Process for writing backtests to DB with a queue

    Args:
        queue (mp.Queue): The queue
        done_flag (_type_): The done flag to indicate when to stop
    """
    while True:
        while not queue.empty():
            forex_pair_id, strategy_config_id, timeframe, metrics_df, score = (
                queue.get()
            )
            try:
                # Re-initialize DB connection in this process
                backtest_sqlhelper = BacktestSQLHelper()

                # Insert backtest run
                run_id = backtest_sqlhelper.insert_backtest_run(
                    forex_pair_id, strategy_config_id, timeframe, metrics_df
                )
                logging.info(f"Saved backtest run (id: {run_id})")

                # Insert composite score
                if score is not None:
                    backtest_sqlhelper.insert_composite_score(run_id, score)
                    logging.info(f"Run met thresholds, saved composite score.")

                # Close DB connection
                backtest_sqlhelper.close_connection()
            except Exception as e:
                logging.error(f"Failed to save backtest: {e}")
                winsound.Beep(300, 500)

        if done_flag.value:
            break

        time.sleep(1)


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


def objective(
    trial: optuna.Trial, forex_pair: str, timeframe: str, db_queue: mp.Queue
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
    strategy = strategies.NNFXStrategy(
        atr, baseline, c1, c2, volume, exit, forex_pair, timeframe
    )
    backtester = Backtester(strategy, forex_pair, timeframe, initial_balance=10_000)

    # Skip configuration if already tested
    backtest_sqlhelper = BacktestSQLHelper()
    pair_id = backtest_sqlhelper.get_forex_pair_id(forex_pair)
    strategy_id = backtest_sqlhelper.select_strategy_configuration(
        strategy.NAME, strategy.DESCRIPTION, strategy.PARAMETER_SETTINGS
    )

    manually_pruned = False
    if backtest_sqlhelper.backtest_run_exists(pair_id, strategy_id, timeframe):
        logging.info("Pruned trial due to existing run.")
        manually_pruned = True
        raise optuna.exceptions.TrialPruned()

    backtest_sqlhelper.close_connection()
    backtest_sqlhelper = None

    score = None
    run_metrics_df = None
    try:
        # Run backtest
        backtester.run_backtest()

        # Retrieve metrics
        run_metrics_df = backtester.get_metrics_df()

        # Prune trials with too few trades to minimize exploring inactive configurations
        min_trades_per_day = config.MIN_TRADES_PER_DAY[timeframe]
        trades_per_day = backtester.get_metric("Trades_Per_Day")
        if trades_per_day < min_trades_per_day:
            logging.info(
                f"Pruned trial due to low activity: {trades_per_day:.2f} trades/day < required {min_trades_per_day} trades/day for {timeframe} timeframe"
            )
            manually_pruned = True
            raise optuna.exceptions.TrialPruned()

        # Calculate composite score
        score = backtester.calculate_composite_score()

        # Replace inf or NaN with 0
        if score is None or score != score or score in (float("inf"), float("-inf")):
            logging.warning("Pruned trial due to invalid score.")
            manually_pruned = True
            score = None
            raise optuna.exceptions.TrialPruned()

        # Optimize for higher score
        return score
    except Exception as e:
        if manually_pruned:
            raise e
        full_traceback = traceback.format_exc()
        logging.error(f"Fatal error on {forex_pair} {timeframe}: {full_traceback}")
        sys.exit(1)
    finally:
        # Send backtester metrics to queue to save results
        if run_metrics_df is not None:
            queue_tuple = (pair_id, strategy_id, timeframe, run_metrics_df, score)
            db_queue.put(queue_tuple)


def run_study_wrapper(args):
    """Wrapper for multiprocessing

    Args:
        args (Tuple): A tuple of (pair, timeframe, n_trials)
    Returns:
        str: The study name
    """
    pair, timeframe, trials, db_queue = args
    logging.info(f"[START STUDY] {pair} @ {timeframe}")

    def wrapped_objective(trial):
        return objective(trial, pair, timeframe, db_queue)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{pair.replace('/', '')}_{timeframe}_NNFX",
        sampler=TPESampler(
            seed=42, n_startup_trials=20
        ),  # n_startup_trials indicates the number of initial random trials
        storage="sqlite:///backtesting/optuna_studies.db",
        load_if_exists=True,
    )

    # Start studying
    study.optimize(wrapped_objective, n_trials=trials, n_jobs=1)

    report_path = f"backtesting/optuna_reports/{pair.replace('/', '')}_{timeframe}"
    save_optuna_report(study, report_path)

    logging.info(f"[FINISHED STUDY] {pair} @ {timeframe}")
    return study.study_name


def run_all_studies(n_trials_per_combo: int = 100, max_parallel_studies: int = 1):
    """Run Optuna studies on all pair/timeframe combos

    Args:
        n_trials_per_combo (int, optional): The number of trials per pair/timeframe combo. Defaults to 100.
        max_parallel_studies (int, optional): The maximum number of parallel studies. Defaults to 1.
    """
    manager = mp.Manager()
    db_queue = manager.Queue()
    done_flag = mp.Value("b", False)

    # Start DB writer process
    writer_proc = mp.Process(target=db_writer_process, args=(db_queue, done_flag))
    writer_proc.start()

    try:
        # Prepare study jobs
        study_args = [
            (pair, timeframe, n_trials_per_combo, db_queue)
            for pair in MAJOR_FOREX_PAIRS
            for timeframe in TIMEFRAMES
        ]

        with ProcessPoolExecutor(max_workers=max_parallel_studies) as executor:
            for study_name in executor.map(run_study_wrapper, study_args):
                logging.info(f"Finished: {study_name}")
    finally:
        done_flag.value = True
        writer_proc.join()


if __name__ == "__main__":
    # Run Optuna studies on all pair/timeframe combos
    num_processes = mp.cpu_count()
    num_processes = 1
    run_all_studies(n_trials_per_combo=300, max_parallel_studies=num_processes)

    # test_one()
