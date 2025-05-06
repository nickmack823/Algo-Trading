import logging
import time
from itertools import product

from scripts import polygon, strategies
from scripts.backtester import Backtester
from scripts.config import MAJOR_FOREX_PAIRS, TIMEFRAMES
from scripts.indicators import (
    baseline_indicators,
    momentum_indicators,
    trend_indicators,
    volatility_indicators,
    volume_indicators,
)
from scripts.sql import BacktestSQLHelper

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)


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


def test_one():
    strategy = strategies.NNFXStrategy(
        atr=volatility_indicators[0],
        baseline=baseline_indicators[0],
        c1=trend_indicators[2],
        c2=trend_indicators[3],
        volume=volume_indicators[0],
        exit_indicator=trend_indicators[4],
        forex_pair="EURUSD",
    )

    logging.info(f"Current strategy: {strategy}")

    backtester = Backtester(strategy, "EURUSD", "15_minute", initial_balance=10_000)

    # Time the backtest
    start_time = time.time()

    # cProfile.run("backtester.run_backtest()", sort="cumtime")
    backtester.run_backtest()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Backtest completed in {elapsed_time:.2f} seconds")

    backtester.save_run()

    trades = backtester.save_trades()


def nnfx_testing():
    # Assume you’ve already defined these lists
    volatility_pool = volatility_indicators
    baseline_pool = baseline_indicators
    confirmation_pool = trend_indicators + momentum_indicators
    volume_pool = volume_indicators
    exit_pool = (
        confirmation_pool  # or use a separate exit_indicators list if you have one
    )

    logging.info(f"Number of volatility indicators: {len(volatility_pool)}")
    logging.info(f"Number of baseline indicators: {len(baseline_pool)}")
    logging.info(f"Number of confirmation indicators: {len(confirmation_pool)}")
    logging.info(f"Number of volume indicators: {len(volume_pool)}")
    logging.info(f"Number of exit indicators: {len(exit_pool)}")

    backtest_sqlhelper = BacktestSQLHelper()

    # Create a generator to lazily produce combinations
    combo_generator = product(
        volatility_pool,
        baseline_pool,
        confirmation_pool,
        confirmation_pool,
        volume_pool,
        exit_pool,
    )

    total_checked = 0
    total_run = 0

    estimated_total_checks = count_valid_combos(
        vol_pool=volatility_pool,
        base_pool=baseline_pool,
        conf_pool=confirmation_pool,
        volu_pool=volume_pool,
        exit_pool=exit_pool,
    )

    logging.info(f"Estimated total checks: {estimated_total_checks}")

    for atr, baseline, c1, c2, volume, exit in combo_generator:
        if c1 == c2:
            continue  # Skip invalid C1 == C2

        for forex_pair in MAJOR_FOREX_PAIRS:
            pair_id = backtest_sqlhelper.get_forex_pair_id(forex_pair)

            for timeframe in TIMEFRAMES:
                # Build strategy
                strategy = strategies.NNFXStrategy(
                    atr=atr,
                    baseline=baseline,
                    c1=c1,
                    c2=c2,
                    volume=volume,
                    exit_indicator=exit,
                    forex_pair=forex_pair,
                )

                strategy_id = backtest_sqlhelper.select_strategy_configuration(
                    strategy.NAME, strategy.DESCRIPTION, strategy.PARAMETER_SETTINGS
                )

                run_exists = backtest_sqlhelper.backtest_run_exists(
                    pair_id, strategy_id, timeframe
                )

                if not run_exists:
                    backtester = Backtester(
                        strategy,
                        forex_pair.replace("/", ""),
                        timeframe,
                        initial_balance=10_000,
                    )
                    backtester.run_backtest()
                    backtester.save_run()
                    total_runs += 1
                else:
                    logging.debug(
                        f"Already exists: {strategy.NAME} | {forex_pair} | {timeframe}"
                    )

        total_checked += 1

        # Optional: stop early for testing
        # if total_run >= 100:
        #     break

    logging.info(f"Total combinations checked: {total_checked}")
    logging.info(f"Total new backtests run: {total_run}")


if __name__ == "__main__":
    # Collect any historical data we don't already have
    # polygon.collect_all_data()

    # Test one strategy
    # test_one()

    # Run backtests
    nnfx_testing()
