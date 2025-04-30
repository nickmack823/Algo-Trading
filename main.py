import cProfile
import logging
from itertools import product

import scripts.sql as sql
from scripts import polygon, strategies
from scripts.backtester import Backtester
from scripts.indicators import (
    baseline_indicators,
    momentum_indicators,
    trend_indicators,
    volatility_indicators,
    volume_indicators,
)

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)


def nnfx_testing():
    # Assume you’ve already defined these lists
    volatility_pool = volatility_indicators
    baseline_pool = baseline_indicators
    confirmation_pool = trend_indicators + momentum_indicators
    volume_pool = volume_indicators
    exit_pool = (
        confirmation_pool  # or use a separate exit_indicators list if you have one
    )

    # Generate all combinations of (atr, baseline, c1, c2, volume, exit)
    all_combos = product(
        volatility_pool,
        baseline_pool,
        confirmation_pool,
        confirmation_pool,
        volume_pool,
        exit_pool,
    )

    # Filter: C1 and C2 must differ
    valid_combos = [
        (atr, baseline, c1, c2, volume, exit)
        for (atr, baseline, c1, c2, volume, exit) in all_combos
        if c1 != c2
    ]

    for atr, baseline, c1, c2, volume, exit in valid_combos:
        strategy = strategies.NNFXStrategy(
            atr=atr,
            baseline=baseline,
            c1=c1,
            c2=c2,
            volume=volume,
            exit_indicator=exit,
            forex_pair="EURUSD",
        )

        backtester = Backtester(strategy, "EURUSD", "1_day", initial_balance=10_000)
        backtester.run_backtest()


if __name__ == "__main__":
    # Collect any historical data we don't already have
    # polygon.collect_all_data()

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

    backtester = Backtester(strategy, "EURUSD", "1_day", initial_balance=10_000)

    # Time the backtest
    import time

    start_time = time.time()

    # cProfile.run("backtester.run_backtest()", sort="cumtime")
    backtester.run_backtest()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Backtest completed in {elapsed_time:.2f} seconds")

    backtester.save_run()

    trades = backtester.save_trades()

    # Run backtest
    # nnfx_testing()
