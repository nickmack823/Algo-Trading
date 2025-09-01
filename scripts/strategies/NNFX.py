import pandas as pd

from scripts.config import (
    BEARISH_SIGNAL,
    BEARISH_TREND,
    BULLISH_SIGNAL,
    BULLISH_TREND,
    DEFAULT_ATR_SL_MULTIPLIER,
    DEFAULT_TP_MULTIPLIER,
    ENTER_LONG,
    ENTER_SHORT,
    EXIT_LONG,
    EXIT_SHORT,
    HIGH_VOLUME,
    NO_SIGNAL,
    RISK_PER_TRADE,
)
from scripts.data.sql import IndicatorCacheSQLHelper
from scripts.indicators.indicator_configs import IndicatorConfig
from scripts.strategies.strategy_core import (
    BaseStrategy,
    PositionCalculator,
    PositionManager,
    TradePlan,
    register_strategy,
)
from scripts.utilities import convert_np_to_pd


@register_strategy(
    "NNFX",
    [
        "atr",
        "baseline",
        "c1",
        "c2",
        "volume",
        "exit_indicator",
        "forex_pair",
        "timeframe",
    ],
    description="No Nonsense Forex (NNFX) Strategy based on complete rule set.",
)
class NNFXStrategy(BaseStrategy):
    NAME = "NNFX"
    DESCRIPTION = "No Nonsense Forex (NNFX) Strategy based on complete rule set."

    def __init__(
        self,
        atr: IndicatorConfig,
        baseline: IndicatorConfig,
        c1: IndicatorConfig,
        c2: IndicatorConfig,
        volume: IndicatorConfig,
        exit_indicator: IndicatorConfig,
        forex_pair: str,
        timeframe: str,
    ):

        self.atr = atr
        self.baseline = baseline
        self.c1 = c1
        self.c2 = c2
        self.volume = volume
        self.exit = exit_indicator

        self.indicator_configs = {
            "ATR": atr,
            "Baseline": baseline,
            "C1": c1,
            "C2": c2,
            "VolumeIndicator": volume,
            "Exit": exit_indicator,
        }
        self.indicator_cache_dicts_to_insert = []

        parameters = {
            "atr": f"{atr['name']}_{atr['parameters']}",
            "baseline": f"{baseline['name']}_{baseline['parameters']}",
            "c1": f"{c1['name']}_{c1['parameters']}",
            "c2": f"{c2['name']}_{c2['parameters']}",
            "volume": f"{volume['name']}_{volume['parameters']}",
            "exit": f"{exit_indicator['name']}_{exit_indicator['parameters']}",
        }

        # For Continuation Entry
        self.last_valid_entry_index = None
        self.last_valid_entry_direction = None

        # For backtesting
        self.data_with_indicators = None

        super().__init__(
            forex_pair=forex_pair, parameters=parameters, timeframe=timeframe
        )

    def prepare_data(self, historical_data: pd.DataFrame, use_cache: bool = True):
        if use_cache:
            self.data_with_indicators = self._calculate_or_retrieve_indicators(
                historical_data
            )
        else:
            self.data_with_indicators = self._calculate_indicators_no_cache(
                historical_data
            )

    def get_cache_jobs(self):
        return self.indicator_cache_dicts_to_insert

    def _calculate_or_retrieve_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        cache = IndicatorCacheSQLHelper()

        data_start_date, data_end_date = (
            data["Timestamp"].min(),
            data["Timestamp"].max(),
        )

        def retrieve_or_calculate(
            config: IndicatorConfig,
        ) -> tuple[pd.DataFrame | pd.Series, bool]:
            calculated = False

            cached_df = cache.fetch(
                config["name"],
                config["parameters"],
                self.FOREX_PAIR,
                self.TIMEFRAME,
                data_start_date,
                data_end_date,
            )
            if cached_df is not None:
                return cached_df, calculated

            raw_output = config["function"](data, **config["parameters"])
            output_df_or_series = convert_np_to_pd(raw_output)
            calculated = True

            return output_df_or_series, calculated

        # Calculate/retrieve cached indicators and add to data DataFrame
        indicator_dfs = []
        for indicator_config in self.indicator_configs.items():
            piece_name, config = indicator_config

            # Retrieve or calculate indicator DataFrame
            output_df_or_series, calculated = retrieve_or_calculate(config)

            if isinstance(output_df_or_series, pd.DataFrame):
                df = output_df_or_series
            else:
                df = pd.DataFrame({config["name"]: output_df_or_series})
            # Ensure columns are strings
            df.columns = df.columns.astype(str)

            # Ensure no duplicate cache jobs (situations where same indicator is for 2+ pieces)
            indicator_in_cache_jobs = any(
                [
                    config["name"] == item["indicator_name"]
                    for item in self.indicator_cache_dicts_to_insert
                ]
            )

            # If we calculated the indicator, store it in feather and get meta to link in DB later
            if calculated and not indicator_in_cache_jobs:
                dict_to_insert = cache.store_in_feather(
                    df,
                    config["name"],
                    config["parameters"],
                    self.FOREX_PAIR,
                    self.TIMEFRAME,
                    data_start_date,
                    data_end_date,
                )

                self.indicator_cache_dicts_to_insert.append(dict_to_insert)

            # Add prefix for later calculations
            df = df.add_prefix(f"{piece_name}_")

            indicator_dfs.append(df)

        # Concatenate once to avoid fragmentation
        indicator_data = pd.concat(indicator_dfs, axis=1)

        # Add indicators to data
        full_data = pd.concat(
            [data.reset_index(drop=True), indicator_data.reset_index(drop=True)], axis=1
        )

        # Close DB connection
        cache.close_connection()
        del cache

        return full_data

    def _calculate_indicators_no_cache(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes indicator values directly from data with no caching or retrieval.
        """
        data = data.copy()

        # Calculate indicators
        indicator_dfs = []
        for piece_name, config in self.indicator_configs.items():
            raw_output = config["function"](data, **config["parameters"])
            output_df_or_series = convert_np_to_pd(raw_output)

            # Wrap as DataFrame and ensure column names
            if isinstance(output_df_or_series, pd.DataFrame):
                df = output_df_or_series
            else:
                df = pd.DataFrame({config["name"]: output_df_or_series})
            df.columns = df.columns.astype(str)

            # Add prefix for distinction
            df = df.add_prefix(f"{piece_name}_")
            indicator_dfs.append(df)

        # Combine original price data and indicators
        indicator_data = pd.concat(indicator_dfs, axis=1)
        full_data = pd.concat(
            [data.reset_index(drop=True), indicator_data.reset_index(drop=True)], axis=1
        )

        return full_data

    def _get_signals_since_last_valid_signal(
        self, close_series: pd.Series, baseline_df_or_series: pd.DataFrame
    ):
        """Get baseline signals since the last valid signal (for Continuation Entry checking)

        Args:
            close_series (pd.Series): A Pandas Series containing the closing prices.
            baseline_df_or_series (pd.DataFrame): A Pandas DataFrame containing the baseline indicator values.

        Returns:
            list[str]: A list of signals
        """
        closes_since_signal = close_series.iloc[self.last_valid_entry_index + 1 :]
        baselines_since_signal = baseline_df_or_series.iloc[
            self.last_valid_entry_index + 1 :
        ]

        signals_since_last_valid_signal = []
        # Generate signals for each data point since the last valid entry
        for i in range(
            1, len(closes_since_signal)
        ):  # Start from the second item (i=1) because we need the previous one
            # Slice the data to include both the previous and current data points
            signal_data_close = closes_since_signal.iloc[i - 1 : i + 1]
            signal_data_baseline = baselines_since_signal.iloc[i - 1 : i + 1]

            # Get the signal for the current data point using the function
            current_signals = self._get_indicator_signals(
                self.baseline["signal_function"],
                signal_data_baseline,
                signal_data_close,
            )

            # Accumulate the signals
            signals_since_last_valid_signal.extend(current_signals)

        return signals_since_last_valid_signal

    def _generate_signals(self, data: pd.DataFrame, current_position: int) -> list[str]:
        """
        Generate NNFX trading signals based on the full strategy logic.

        Args:
            data (pd.DataFrame): Full price history including indicator values.
            current_position (int): 0 (flat), 1 (long), -1 (short)

        Returns:
            list[str]: A list of signals (ENTER_LONG, EXIT_SHORT, etc.)
        """
        signals = []
        signal_source = None

        # Ensure there are at least 10 data points to compare
        if len(data) < 10:
            return [NO_SIGNAL], signal_source

        # Extract current and previous row
        current_row: pd.Series = data.iloc[-1]
        previous_row: pd.Series = data.iloc[-2]

        # Extract ATR, baseline, c1, c2, volume, and exit indicators from data
        atr_df_or_series = self._reconstruct_df(data, "ATR")
        baseline_df_or_series = self._reconstruct_df(data, "Baseline")
        c1_df_or_series = self._reconstruct_df(data, "C1")
        c2_df_or_series = self._reconstruct_df(data, "C2")
        volume_df_or_series = self._reconstruct_df(data, "VolumeIndicator")
        exit_df_or_series = self._reconstruct_df(data, "Exit")

        # Remove tags from column names in DataFrames for passing to signal functions
        if isinstance(atr_df_or_series, pd.DataFrame):
            atr_df_or_series.columns = [
                col.replace("ATR_", "") for col in atr_df_or_series.columns
            ]
        if isinstance(baseline_df_or_series, pd.DataFrame):
            baseline_df_or_series.columns = [
                col.replace("Baseline_", "") for col in baseline_df_or_series.columns
            ]
        if isinstance(c1_df_or_series, pd.DataFrame):
            c1_df_or_series.columns = [
                col.replace("C1_", "") for col in c1_df_or_series.columns
            ]
        if isinstance(c2_df_or_series, pd.DataFrame):
            c2_df_or_series.columns = [
                col.replace("C2_", "") for col in c2_df_or_series.columns
            ]
        if isinstance(volume_df_or_series, pd.DataFrame):
            volume_df_or_series.columns = [
                col.replace("VolumeIndicator_", "")
                for col in volume_df_or_series.columns
            ]
        if isinstance(exit_df_or_series, pd.DataFrame):
            exit_df_or_series.columns = [
                col.replace("Exit_", "") for col in exit_df_or_series.columns
            ]

        close_series: pd.Series = data["Close"]
        prev_close_series: pd.Series = close_series.iloc[:-1]

        # Evaluate signal functions
        # atr_signals = self._get_indicator_signals(
        #     self.atr["signal_function"], atr_series, close_series
        # )
        baseline_signals = self._get_indicator_signals(
            self.baseline["signal_function"], baseline_df_or_series, close_series
        )
        c1_signals = self._get_indicator_signals(
            self.c1["signal_function"], c1_df_or_series, close_series
        )
        c2_signals = self._get_indicator_signals(
            self.c2["signal_function"], c2_df_or_series, close_series
        )
        volume_signals = self._get_indicator_signals(
            self.volume["signal_function"], volume_df_or_series, close_series
        )
        exit_signals = self._get_indicator_signals(
            self.exit["signal_function"], exit_df_or_series, close_series
        )

        # Evaluate previous candle signals
        baseline_signals_prev = self._get_indicator_signals(
            self.baseline["signal_function"],
            baseline_df_or_series.iloc[:-1],
            prev_close_series,
        )

        # Handle case where C1 returns a DataFrame (use the first column)
        if isinstance(c1_df_or_series, pd.DataFrame):
            c1_prev = c1_df_or_series.iloc[:-1, :]
        else:
            c1_prev = c1_df_or_series.iloc[:-1]
        c1_signals_prev = self._get_indicator_signals(
            self.c1["signal_function"],
            c1_prev,
            prev_close_series,
        )

        # Extract indicator values
        atr_value: float = (
            atr_df_or_series.iloc[-1]
            if isinstance(atr_df_or_series, pd.Series)
            else atr_df_or_series.iloc[-1, 0]
        )
        prev_atr_value: float = (
            atr_df_or_series.iloc[-2]
            if isinstance(atr_df_or_series, pd.Series)
            else atr_df_or_series.iloc[-2, 0]
        )
        baseline_value: float = (
            baseline_df_or_series.iloc[-1]
            if isinstance(baseline_df_or_series, pd.Series)
            else baseline_df_or_series.iloc[-1, 0]
        )
        prev_baseline_value: float = (
            baseline_df_or_series.iloc[-2]
            if isinstance(baseline_df_or_series, pd.Series)
            else baseline_df_or_series.iloc[-2, 0]
        )

        # Check conditions
        price_within_1x_atr_of_baseline = (
            abs(current_row["Close"] - baseline_value) < 1 * atr_value
        )
        prev_price_within_1x_atr_of_baseline = (
            abs(previous_row["Close"] - prev_baseline_value) < 1 * prev_atr_value
        )

        baseline_bullish_signal, baseline_bearish_signal = (
            BULLISH_SIGNAL in baseline_signals,
            BEARISH_SIGNAL in baseline_signals,
        )
        baseline_bullish_trend, baseline_bearish_trend = (
            BULLISH_TREND in baseline_signals,
            BEARISH_TREND in baseline_signals,
        )
        prev_baseline_bullish_signal, prev_baseline_bearish_signal = (
            BULLISH_SIGNAL in baseline_signals_prev,
            BEARISH_SIGNAL in baseline_signals_prev,
        )

        c1_bullish_signal, c1_bearish_signal = (
            BULLISH_SIGNAL in c1_signals,
            BEARISH_SIGNAL in c1_signals,
        )
        c1_bullish_trend, c1_bearish_trend = (
            BULLISH_TREND in c1_signals,
            BEARISH_TREND in c1_signals,
        )

        prev_c1_bullish_trend, prev_c1_bearish_trend = (
            BULLISH_TREND in c1_signals_prev,
            BEARISH_TREND in c1_signals_prev,
        )
        c2_bullish_signal, c2_bearish_signal = (
            BULLISH_SIGNAL in c2_signals,
            BEARISH_SIGNAL in c2_signals,
        )
        c2_bullish_trend, c2_bearish_trend = (
            BULLISH_TREND in c2_signals,
            BEARISH_TREND in c2_signals,
        )

        volume_high = HIGH_VOLUME in volume_signals

        exit_bullish_signal, exit_bearish_signal = (
            BULLISH_SIGNAL in exit_signals,
            BEARISH_SIGNAL in exit_signals,
        )

        # --- Find most recent C1 signal within 7 candles (7-Candle Rule for Baseline Cross Entry) ---
        recent_c1_signals = []

        # Loop over candles -2 to -7 (inclusive)
        for offset in range(2, 8):  # candles ago
            if isinstance(c1_df_or_series, pd.DataFrame):
                partial_output = c1_df_or_series.iloc[: -offset + 1, :]
            else:
                partial_output = c1_df_or_series.iloc[: -offset + 1]

            signals = self._get_indicator_signals(
                self.c1["signal_function"], partial_output, close_series
            )
            recent_c1_signals.append(signals)

        # Flatten the signals
        # flattened_signals = [sig for sublist in recent_c1_signals for sig in sublist]

        # Evaluate 7 Candle Rule
        # seven_candle_c1_bullish = BULLISH_SIGNAL in flattened_signals
        # seven_candle_c1_bearish = BEARISH_SIGNAL in flattened_signals

        # Signal dict for testing
        signal_dict = {
            "baseline_bullish_signal": baseline_bullish_signal,
            "baseline_bearish_signal": baseline_bearish_signal,
            "baseline_bullish_trend": baseline_bullish_trend,
            "baseline_bearish_trend": baseline_bearish_trend,
            "c1_bullish_signal": c1_bullish_signal,
            "c1_bearish_signal": c1_bearish_signal,
            "c1_bullish_trend": c1_bullish_trend,
            "c1_bearish_trend": c1_bearish_trend,
            "c2_bullish_signal": c2_bullish_signal,
            "c2_bearish_signal": c2_bearish_signal,
            "c2_bullish_trend": c2_bullish_trend,
            "c2_bearish_trend": c2_bearish_trend,
            "volume_high": volume_high,
            "exit_bullish_signal": exit_bullish_signal,
            "exit_bearish_signal": exit_bearish_signal,
            "price_within_1x_atr_of_baseline": price_within_1x_atr_of_baseline,
            # "seven_candle_c1_bullish": seven_candle_c1_bullish,
            # "seven_candle_c1_bearish": seven_candle_c1_bearish,
        }

        # --- ENTRY CONDITIONS ---
        if current_position == 0:
            # 1. Standard Entry - C1 gives signal & others agree
            if (
                c1_bullish_signal
                and baseline_bullish_trend
                and price_within_1x_atr_of_baseline
                and c2_bullish_trend
                and volume_high
            ):
                signals.append(ENTER_LONG)
                signal_source = "Standard Entry"
            elif (
                c1_bearish_signal
                and baseline_bearish_trend
                and price_within_1x_atr_of_baseline
                and c2_bearish_trend
                and volume_high
            ):
                signals.append(ENTER_SHORT)
                signal_source = "Standard Entry"

            # 2. Baseline Cross Entry - Baseline gives signal (price crosses over) & others agree
            elif (
                baseline_bullish_signal
                # and seven_candle_c1_bullish
                and c1_bullish_trend
                and price_within_1x_atr_of_baseline
                and c2_bullish_trend
                and volume_high
            ):
                signals.append(ENTER_LONG)
                signal_source = "Baseline Cross Entry"
            elif (
                baseline_bearish_signal
                # and seven_candle_c1_bearish
                and c1_bearish_trend
                and price_within_1x_atr_of_baseline
                and c2_bearish_trend
                and volume_high
            ):
                signals.append(ENTER_SHORT)
                signal_source = "Baseline Cross Entry"

            # 3. Pullback Entry - Baseline gives signal (price crosses over) on previous candle shooting beyond 1x ATR,
            # current candle pulls back to within 1x ATR & others agree
            elif (
                prev_baseline_bullish_signal
                and prev_c1_bullish_trend
                and not prev_price_within_1x_atr_of_baseline
            ):
                # Check for pullback after strong cross above baseline
                if (
                    c1_bullish_trend
                    and price_within_1x_atr_of_baseline
                    and c2_bullish_trend
                    and volume_high
                ):
                    signals.append(ENTER_LONG)
                    signal_source = "Pullback Entry"
            elif (
                prev_baseline_bearish_signal
                and prev_c1_bearish_trend
                and not prev_price_within_1x_atr_of_baseline
            ):
                # Check for pullback after strong cross below baseline
                if (
                    c1_bearish_trend
                    and price_within_1x_atr_of_baseline
                    and c2_bearish_trend
                    and volume_high
                ):
                    signals.append(ENTER_SHORT)
                    signal_source = "Pullback Entry"

            # 4. Continuation Entry - 1 of the 3 entry signals happened on a recent previous candle
            # regardless of if we entered, price has not crossed baseline in opposite direction since then,
            # and we receive another signal
            elif (
                self.last_valid_entry_index is not None
                and self.last_valid_entry_direction == "Long"
            ):
                # Has price crossed ABOVE the baseline since the last valid signal?
                below_baseline_since_signal = (
                    BEARISH_SIGNAL
                    in self._get_signals_since_last_valid_signal(
                        close_series, baseline_df_or_series
                    )
                )

                # If not crossed and current signals align, enter again
                if (
                    not below_baseline_since_signal
                    and c1_bullish_signal
                    and c2_bullish_trend
                ):  # and volume_high:
                    signals.append(ENTER_LONG)
                    signal_source = "Continuation Entry"

            elif (
                self.last_valid_entry_index is not None
                and self.last_valid_entry_direction == "Short"
            ):
                # Has price crossed ABOVE the baseline since the last valid signal?
                above_baseline_since_signal = (
                    BULLISH_SIGNAL
                    in self._get_signals_since_last_valid_signal(
                        close_series, baseline_df_or_series
                    )
                )

                if (
                    not above_baseline_since_signal
                    and c1_bearish_signal
                    and c2_bearish_trend
                ):  # and volume_high:
                    signals.append(ENTER_SHORT)
                    signal_source = "Continuation Entry"

        # --- EXIT CONDITIONS ---
        elif current_position == 1:
            if (
                exit_bearish_signal
                or baseline_bearish_signal
                # or c1_bearish_signal
                # or c2_bearish_signal
            ):
                signals.append(EXIT_LONG)
                signal_source = "Exit"

        elif current_position == -1:
            if (
                exit_bullish_signal
                or baseline_bullish_signal
                # or c1_bullish_signal
                # or c2_bullish_signal
            ):
                signals.append(EXIT_SHORT)
                signal_source = "Exit"

        if not signals:
            signals.append(NO_SIGNAL)

        # Update the last valid entry index and direction for Continuation Entry
        if ENTER_LONG in signals or ENTER_SHORT in signals:
            self.last_valid_entry_index = len(data) - 1
            self.last_valid_entry_direction = (
                "Long" if ENTER_LONG in signals else "Short"
            )

        return signals, signal_source

    def _get_piece_latest_value(self, piece_name: str, row: pd.Series) -> float:
        """Find the column with the given piece_name as a prefix, then return the value of that column"""
        for col in row.index:
            if col.startswith(f"{piece_name}_"):
                return row[col]

        raise KeyError(f"No column found with prefix '{piece_name}'")

    # Main trade planning function to be called during each backtest iteration
    # Determines whether to enter, exit, or do nothing based on signals and market context
    def generate_trade_plan(
        self,
        current_index: int,
        current_position: int,
        balance: float,
        quote_to_usd_rate: float,
    ) -> list[TradePlan]:
        """
        Main entry point for backtester. Evaluates latest row of data and current position,
        then returns zero or more StrategyPlans depending on signal type.
        """
        trade_plans = []

        # Step 1: Get current set of data with indicators
        data_with_indicators = self._get_slice(current_index)
        row = data_with_indicators.iloc[-1]

        # Step 2: Generate entry/exit signals using internal logic
        signals, signal_source = self._generate_signals(
            data_with_indicators, current_position
        )

        # Check Entry signals
        if ENTER_LONG in signals or ENTER_SHORT in signals:
            # Step 3: Determine direction of entry signal
            signal = ENTER_LONG if ENTER_LONG in signals else ENTER_SHORT
            atr = self._get_piece_latest_value("ATR", row)

            # if math.isnan(atr):
            #     x = 1

            entry_price = row["Close"]
            direction = "BUY" if signal == ENTER_LONG else "SELL"

            # Step 4: Create two Trades — one with fixed TP (TP1), one as runner
            entries = [
                (RISK_PER_TRADE, DEFAULT_TP_MULTIPLIER, "TP1"),
                (RISK_PER_TRADE, None, "Runner"),
            ]

            for risk_pct, tp_multiplier, tag in entries:
                # Step 5: Calculate stop loss at 1.5x ATR from entry
                sl_price = (
                    entry_price - atr * DEFAULT_ATR_SL_MULTIPLIER
                    if direction == "BUY"
                    else entry_price + atr * DEFAULT_ATR_SL_MULTIPLIER
                )

                # Step 6: Calculate take profit at 1x ATR for TP1; None for runner
                tp_price = (
                    entry_price + atr * DEFAULT_TP_MULTIPLIER
                    if direction == "BUY" and tp_multiplier
                    else (
                        entry_price - atr * DEFAULT_TP_MULTIPLIER
                        if direction == "SELL" and tp_multiplier
                        else None
                    )
                )

                # Step 7: Calculate units to trade based on stop loss size and risk %
                units = PositionCalculator.calculate_trade_units(
                    self.FOREX_PAIR,
                    balance,
                    risk_pct,
                    entry_price,
                    atr,
                    quote_to_usd_rate,
                )

                # Step 8: Create TradePlan with all necessary parameters
                entry_trade_plan = TradePlan(
                    strategy=self.NAME,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    units=units,
                    risk_pct=risk_pct,
                    tag=tag,
                    source=signal_source,
                )

                # Step 9: If this is the Runner, attach a PositionManager to handle breakeven + trailing SL
                runner_config = {}
                if tag == "Runner":
                    runner_config = {
                        "breakeven_trigger": atr,  # move SL to BE at entry price +- 1xATR
                        "trail_start": 2 * atr,  # begin trailing at 2xATR
                        "trail_distance": 1.5 * atr,  # SL trails at 1.5xATR behind
                    }
                entry_trade_plan.position_manager = PositionManager(
                    entry_price=entry_price,
                    initial_stop_loss=sl_price,
                    direction=direction,
                    config=runner_config,
                    take_profit=tp_price,
                )
                trade_plans.append(entry_trade_plan)

        # Step 10. Check for Exit signals
        if EXIT_LONG in signals or EXIT_SHORT in signals:
            # Return a StrategyPlan with only direction and tag to indicate exit intent
            direction = "SELL" if EXIT_LONG in signals else "BUY"

            # Create a single exit signal plan to be handled by the backtester
            exit_plan = TradePlan(
                strategy=self.NAME,
                direction=direction,
                entry_price=row["Close"],
                stop_loss=None,
                tag="EXIT",
                source=signal_source,
            )

            trade_plans.append(exit_plan)

        return trade_plans
