import datetime
import gc
import json
import logging
import multiprocessing as mp
import time
import traceback
import warnings
import winsound
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, stdev

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from scripts.backtester import Backtester
from scripts.config import (
    ALL_TIMEFRAMES,
    DATA_FOLDER,
    MAJOR_FOREX_PAIRS,
    N_STARTUP_TRIALS_PERCENTAGE,
    NNFX_TIMEFRAMES,
    OPTUNA_STUDIES_FOLDER,
    PHASE2_TOP_PERCENT,
    TIMEFRAME_DATE_RANGES_PHASE3,
    TIMEFRAME_DATE_RANGES_PHASE_1_AND_2,
)
from scripts.data.sql import (
    BacktestSQLHelper,
    HistoricalDataSQLHelper,
    IndicatorCacheSQLHelper,
)
from scripts.strategies import candlestick_strategy  # to register them in st
from scripts.strategies import nnfx_strategy, strategy_core
from scripts.trial_adapters import candlestick_adapter, nnxf_adapter
from scripts.trial_adapters.base import ADAPTER_REGISTRY, Acknowledgement, objective

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)


# Suppress all FutureWarnings (warnings of deprecation in future package versions)
warnings.simplefilter(action="ignore", category=FutureWarning)


def log_error(message: str, filename: str = "error_log.txt"):
    """Appends a timestamped error message to a file."""
    winsound.Beep(300, 500)
    logging.error(message)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(log_line)


def run_fixed_strategy_evaluation(
    strategy: strategy_core.BaseStrategy,
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
        TIMEFRAME_DATE_RANGES_PHASE3[timeframe]["from_date"],
        TIMEFRAME_DATE_RANGES_PHASE3[timeframe]["to_date"],
    )

    study_name = f"{phase_name}_{pair.replace('/', '')}_{timeframe}_{strategy.NAME}_{from_date}to{to_date}_space-{exploration_space}"

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


def db_writer_process(queue: mp.Queue, done_flag, ack_dict: dict):
    sql = BacktestSQLHelper()
    cache_sql = IndicatorCacheSQLHelper()

    while True:
        while not queue.empty():
            queue_dict: dict = queue.get(timeout=1)

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
                        ack_dict[queue_dict["ack_id"]] = Acknowledgement(
                            ok=True, payload={"strategy_config_id": strategy_config_id}
                        )

                elif purpose == "indicator_cache":
                    cache_items: list[dict] = queue_dict.get("cache_items")
                    cache_sql.insert_cache_items(cache_items)

                # Write trial to DB (backtest & score || score)
                elif purpose == "trial":
                    # Unpack queue item
                    pair_id = queue_dict.get("pair_id")
                    strategy: strategy_core.NNFXStrategy = queue_dict.get("strategy")
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
                if (
                    purpose == "strategy_config"
                    and ack_dict is not None
                    and queue_dict.get("ack_id")
                ):
                    ack_dict[queue_dict["ack_id"]] = Acknowledgement(
                        ok=False, error=str(e)
                    )
                log_error(f"DB write error: {e} {traceback.format_exc()}")

        if done_flag.value is True:
            # Close DB connection
            logging.info("Closing DB queue connection...")
            sql.close_connection()
            cache_sql.close_connection()
            del sql
            del cache_sql
            break


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
        strategy_key,  # <-- ADD
        context,  # <-- ADD (adapter-specific payload)
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
            strategy_key=strategy_key,
            context=context,
        )

    timeframe_range = TIMEFRAME_DATE_RANGES_PHASE_1_AND_2[timeframe]
    from_date, to_date = timeframe_range["from_date"], timeframe_range["to_date"]

    # callback_n = IMPROVEMENT_CUTOFF_BY_TIMEFRAME[timeframe]

    study_name = (
        f"{phase_name}_{pair.replace('/', '')}_{timeframe}_{strategy_key}_"
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
    valid_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    completed_or_decided = [t for t in study.trials if t.state in valid_states]
    remaining_trials = max(n_trials - len(completed_or_decided), 0)
    logging.info(
        f"[{study_name}]: finished={len(completed_or_decided)} to_run={remaining_trials}"
    )

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
                    error_msg = f"[ERROR] Study failed - Error: {e} | Traceback: {traceback.format_exc()}"
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
    args: list[tuple] = []
    for key in STRATEGY_KEYS_TO_RUN:
        adapter = ADAPTER_REGISTRY[key]
        args.extend(
            adapter.build_phase1_args(
                pairs=pairs,
                timeframes=timeframes,
                seeds=seeds,
                trials_by_timeframe=trials_by_timeframe,
                exploration_space=exploration_space,
                phase_name=phase_name,
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
    args: list[tuple] = []
    for key in STRATEGY_KEYS_TO_RUN:
        adapter = ADAPTER_REGISTRY[key]
        args.extend(
            adapter.build_phase2_args(
                pairs=pairs,
                timeframes=timeframes,
                seeds=seeds,
                trials_by_timeframe=trials_by_timeframe,
                top_percent=top_percent,
                exploration_space=exploration_space,
                phase_name=phase_name,
            )
        )
    return args


def build_study_args_phase3(
    pairs: list[str],
    timeframes: list[str],
    top_n: int,
) -> list[tuple]:
    args: list[tuple] = []
    for key in STRATEGY_KEYS_TO_RUN:
        adapter = ADAPTER_REGISTRY[key]
        args.extend(
            adapter.build_phase3_args(
                pairs=pairs,
                timeframes=timeframes,
                top_n=top_n,
            )
        )
    return args


def run_phase3(phase: dict):
    # Allow phase3 to honor a custom timeframe list; otherwise take all
    timeframes = phase.get("timeframes", ALL_TIMEFRAMES)

    run_args = build_study_args_phase3(
        pairs=MAJOR_FOREX_PAIRS,
        timeframes=timeframes,
        top_n=phase["top_n"],
    )

    logging.info(
        f"Running {len(run_args)} fixed strategy evaluations across all pair/timeframe combos..."
    )

    with mp.Manager() as manager:
        db_queue = manager.Queue()
        ack_dict = manager.dict()
        done_flag = manager.Value("b", False)

        writer_proc = mp.Process(
            target=db_writer_process, args=(db_queue, done_flag, ack_dict)
        )
        writer_proc.start()

        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
                futures = [
                    executor.submit(run_fixed_strategy_evaluation, *args, db_queue)
                    for args in run_args
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                        completed += 1
                        logging.info(f"Progress: {completed}/{len(run_args)} completed")
                    except Exception as e:
                        logging.error(f"[ERROR] Phase 3 job failed: {e}")
        finally:
            done_flag.value = True
            writer_proc.join()

    logging.info("Phase 3 complete.")
    return


if __name__ == "__main__":
    # --- Edit these top-level knobs freely ---
    STRATEGY_KEYS_TO_RUN: list[str] = [
        # "NNFX",
        "CANDLESTICK",
    ]  # use any registered adapters
    N_PROCESSES: int = 1  # mp.cpu_count() - 3 is a decent default

    # Seeds you want to sweep (phase1/2); pick a set below:
    SEED_SETS: list[list[int]] = [
        [42, 1337, 314, 777, 444],  # rich sweep
        [42, 1337],  # quick sanity
        [823, 145, 420, 666],  # alt sweep
        [823, 145],  # alt quick
    ]
    CURRENT_SEEDS: list[int] = SEED_SETS[0]

    # Timeframes you want the exploration phases to use (kept visible here)
    # You can swap this for ALL_TIMEFRAMES or any custom list.
    TIMEFRAMES_PHASES_1_2: list[str] = NNFX_TIMEFRAMES

    # Single source of truth for trials per timeframe (used by phase1 AND phase2)
    TRIALS_BY_TIMEFRAME: dict[str, int] = {
        "1_day": 500,
        "4_hour": 300,
        "2_hour": 300,
        "1_hour": 200,
        "30_minute": 200,
        "15_minute": 200,
        "5_minute": 200,
    }

    # Phase list—edit here, not in the loop
    PHASES: list[dict] = [
        {
            "name": "phase1",
            "exploration_space": "default",
            "seeds": CURRENT_SEEDS,
            "timeframes": TIMEFRAMES_PHASES_1_2,
            "trials_by_timeframe": TRIALS_BY_TIMEFRAME,
        },
        {
            "name": "phase2",
            "exploration_space": f"top_{PHASE2_TOP_PERCENT}percent_parameterized",
            "seeds": CURRENT_SEEDS,
            "timeframes": TIMEFRAMES_PHASES_1_2,
            "trials_by_timeframe": TRIALS_BY_TIMEFRAME,
            "top_percent": PHASE2_TOP_PERCENT,  # if your adapter uses it
        },
        {
            "name": "phase3",
            # evaluate fixed strategies using best configs discovered earlier
            "top_n": 30,  # change to 3/5/etc if you want more per pair
            # If you want to restrict timeframes for phase3, add:
            # "timeframes": ["4_hour","1_hour","15_minute"]
        },
    ]

    logging.info(f"Running phases {[p['name'] for p in PHASES]}...")

    # Choose pairs once—keep visible here
    pairs = MAJOR_FOREX_PAIRS
    logging.info(f"Using Pairs: {pairs}")

    for phase in PHASES:
        name = phase["name"]
        logging.info(f"Running {name}...")

        if name in ("phase1", "phase2"):
            # Build kwargs for builder
            kwargs = {
                "pairs": pairs,
                "timeframes": phase.get("timeframes", TIMEFRAMES_PHASES_1_2),
                "seeds": phase["seeds"],
                "trials_by_timeframe": phase["trials_by_timeframe"],
                "exploration_space": phase["exploration_space"],
                "phase_name": name,
            }
            logging.info(f"Using Timeframes: {kwargs['timeframes']}")

            if name == "phase1":
                study_args = build_study_args_phase1(**kwargs)
            else:  # phase2
                # If your adapter’s phase2 builder expects top_percent, pass it:
                kwargs["top_percent"] = phase.get("top_percent", PHASE2_TOP_PERCENT)
                study_args = build_study_args_phase2(**kwargs)

            logging.info(
                f"Running {len(study_args)} total jobs across all pair/timeframe/seed combos..."
            )
            run_all_studies(study_args, max_parallel_studies=N_PROCESSES)

        elif name == "phase3":
            run_phase3(phase)
        else:
            raise ValueError(f"Unknown phase name: {name}")

        logging.info(f"{name} complete.")
        time.sleep(5)

    # Persist best results to JSON for easy inspection
    final_best_results = BacktestSQLHelper(
        read_only=True
    ).get_best_strategy_per_pair_with_metrics()
    with open("best_strategies.json", "w", encoding="utf-8") as f:
        json.dump(final_best_results, f, indent=2)
