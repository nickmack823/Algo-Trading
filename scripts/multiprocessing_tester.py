import logging
import multiprocessing as mp
import time
from itertools import product

from scripts import strategies
from scripts.backtester import Backtester
from scripts.config import MAJOR_FOREX_PAIRS, TIMEFRAMES
from scripts.indicators import (
    baseline_indicators,
    momentum_indicators,
    trend_indicators,
    volatility_indicators,
    volume_indicators,
)
from scripts.utilities import seconds_to_dhms_str


def generate_variants_all_numeric_params(
    indicator_def: dict, step: int = 2, range_size: int = 10
) -> list[dict]:
    """
    Generate all combinations of parameter variants for every numeric parameter in the indicator.
    """
    if not indicator_def.get("parameters"):
        return [indicator_def]

    numeric_params = {
        k: v
        for k, v in indicator_def["parameters"].items()
        if isinstance(v, (int, float))
    }

    if not numeric_params:
        return [indicator_def]

    # Generate list of (param_name, [values])
    param_values_list = []
    for param, base in numeric_params.items():
        values = [base + step * delta for delta in range(-range_size, range_size + 1)]
        values = [v for v in values if v > 0]
        param_values_list.append((param, values))

    # All combinations of parameter values
    all_param_combos = list(product(*[vals for _, vals in param_values_list]))

    variants = []
    for combo in all_param_combos:
        variant = dict(indicator_def)
        variant["parameters"] = dict(indicator_def["parameters"])  # deep copy of params

        name_parts = [indicator_def["name"]]
        for (param, _), value in zip(param_values_list, combo):
            variant["parameters"][param] = value
            name_parts.append(
                f"{param}{int(value) if isinstance(value, int) else round(value, 2)}"
            )

        variant["name"] = "_".join(name_parts)
        variants.append(variant)

    return variants


# Returns parameter combinations of input indicators.
# For use after finding best performers on their default settings
def expand_indicator_pool(indicators: list[dict]) -> list[dict]:
    expanded = []
    for ind in indicators:
        expanded.extend(generate_variants_all_numeric_params(ind))
    return expanded


def count_valid_combos(vol_pool, base_pool, conf_pool, volu_pool, exit_pool):
    total = 0
    for atr, baseline, c1, c2, volume, exit in product(
        vol_pool, base_pool, conf_pool, conf_pool, volu_pool, exit_pool
    ):
        if c1 != c2:
            total += 1
    return total


# Generate strategy + pair/timeframe combinations
def generate_combo_tasks():
    volatility_pool = volatility_indicators
    baseline_pool = baseline_indicators
    confirmation_pool = trend_indicators + momentum_indicators
    volume_pool = volume_indicators
    exit_pool = (
        confirmation_pool  # or use a separate exit_indicators list if you have one
    )

    for atr, baseline, c1, c2, volume, exit in product(
        volatility_pool,
        baseline_pool,
        confirmation_pool,
        confirmation_pool,
        volume_pool,
        exit_pool,
    ):
        if c1 != c2:
            for forex_pair in MAJOR_FOREX_PAIRS:
                for timeframe in TIMEFRAMES:
                    yield (atr, baseline, c1, c2, volume, exit, forex_pair, timeframe)


def estimate_total_tasks():
    n_atr = len(volatility_indicators)
    n_base = len(baseline_indicators)
    n_conf = len(trend_indicators + momentum_indicators)
    n_vol = len(volume_indicators)
    n_exit = n_conf  # or another list if using separate exit indicators

    c1_c2_pairs = n_conf * (n_conf - 1)  # c1 != c2
    total = (
        n_atr
        * n_base
        * c1_c2_pairs
        * n_vol
        * n_exit
        * len(MAJOR_FOREX_PAIRS)
        * len(TIMEFRAMES)
    )
    return total


# Worker builds backtester, runs backtest, returns to queue
def run_combo_worker(task):
    atr, baseline, c1, c2, volume, exit, forex_pair, timeframe = task
    strategy = strategies.NNFXStrategy(
        atr, baseline, c1, c2, volume, exit, forex_pair, timeframe
    )
    backtester = Backtester(
        strategy, forex_pair.replace("/", ""), timeframe, initial_balance=10_000
    )

    # logging.info(
    #     f"Current test: {forex_pair} {timeframe} {backtester.strategy.PARAMETER_SETTINGS}"
    # )

    # Check if this run is already in DB
    pair_id = backtester.backtest_sqlhelper.get_forex_pair_id(forex_pair)
    strategy_id = backtester.backtest_sqlhelper.select_strategy_configuration(
        strategy.NAME, strategy.DESCRIPTION, strategy.PARAMETER_SETTINGS
    )
    run_exists = backtester.backtest_sqlhelper.backtest_run_exists(
        pair_id, strategy_id, timeframe
    )

    if run_exists:
        logging.info(f"Skipping run...")
        backtester = None
    else:
        backtester.run_backtest()

        # Clear SQLHelper connection (it can't be pickled)
        backtester.clear_sqlhelper_connection()

    return backtester


# Single writer process receives backtester objects and saves them
def db_writer(queue: mp.Queue, done_flag):
    while True:
        while not queue.empty():
            backtester: Backtester = queue.get()
            if backtester is None:
                continue
            # logging.info(
            #     f"Saving {backtester.strategy.NAME} {backtester.strategy.FOREX_PAIR} {backtester.strategy.PARAMETER_SETTINGS}"
            # )
            # Reinitialize SQLHelper (it couldn't be pickled, so we closed it earlier)
            backtester.initialize_sqlhelper()
            backtester.save_run()
        if done_flag.value:
            break
        time.sleep(0.1)


# Multiprocessing controller
def parallel_nnfx_testing_with_writer(processes: int):
    start_time = time.time()
    completed = mp.Value("i", 0)
    done_flag = mp.Value("b", False)
    queue = mp.Queue()

    # Estimate total upfront only if needed for progress tracking (optional)
    total_tasks = estimate_total_tasks()
    logging.info(f"Estimated total tasks: {total_tasks}")

    # Start the writer process
    writer_proc = mp.Process(target=db_writer, args=(queue, done_flag))

    logging.info("Starting backtesting...")
    writer_proc.start()

    def callback(backtester):
        queue.put(backtester)
        with completed.get_lock():
            completed.value += 1
            elapsed = time.time() - start_time
            rate = completed.value / elapsed if elapsed > 0 else 0
            eta = (total_tasks - completed.value) / rate if rate > 0 else float("inf")
            logging.info(
                f"\rProgress: {completed.value}/{total_tasks} "
                f"({(completed.value / total_tasks) * 100:.2f}%) | "
                f"Elapsed: {seconds_to_dhms_str(elapsed)} | "
                f"Tests/s: {rate:.2f} | "
                f"ETA: {seconds_to_dhms_str(eta)}",
            )

    with mp.Pool(processes=processes) as pool:
        for result in pool.imap_unordered(
            run_combo_worker, generate_combo_tasks(), chunksize=10
        ):
            callback(result)

    with done_flag.get_lock():
        done_flag.value = True
    writer_proc.join()
    print("\nBacktesting complete.")


# Entry point
if __name__ == "__main__":

    # polygon.collect_all_data()

    processes = round(mp.cpu_count() / 2)
    processes = 1
    # parallel_nnfx_testing_with_writer(processes=processes)
