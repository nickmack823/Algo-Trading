import inspect
from dataclasses import dataclass
from typing import Callable, TypedDict

import numpy as np
import pandas as pd

from scripts.config import *


# Indicator Config for indicator parameters in strategies
class IndicatorConfig(TypedDict):
    name: str
    function: Callable[[pd.DataFrame], pd.Series]
    description: str
    signal_function: Callable
    raw_function: Callable
    parameters: dict


class StrategySignal:
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_SHORT = "EXIT_SHORT"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class TradePlan:
    strategy: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float = None
    trailing_stop: float = None
    units: int = 0
    risk_pct: float = 0.01
    tag: str = ""
    position_manager: "PositionManager" = None
    source: str = (
        ""  # Optional identifier for entry type, e.g., 'StandardEntry', 'Pullback'
    )


class PositionManager:
    """
    Manages stop loss, take profit, breakeven, and trailing logic for a single trade.
    """

    def __init__(
        self,
        entry_price: float,
        initial_stop_loss: float,
        direction: str,
        config: dict[str, float] = None,
        take_profit: float = None,
    ):
        self.entry_price = entry_price
        self.direction = direction.upper()
        self.stop_loss = initial_stop_loss
        self.take_profit = take_profit
        self.config = config or {}
        self.trailing_steps = []
        self.hit_stop_loss = False
        self.hit_take_profit = False

    def update_stop_loss(self, current_price: float) -> float:
        # Breakeven logic
        if self.config.get("breakeven_trigger") is not None:
            breakeven_trigger_diff = self.config["breakeven_trigger"]

            if self.direction == "BUY":
                breakeven_trigger_price = (
                    self.entry_price + breakeven_trigger_diff
                )  # Price where we will move SL to BE
                if current_price >= breakeven_trigger_price:
                    self.stop_loss = max(
                        self.stop_loss, self.entry_price
                    )  # If we cross trigger price, update SL to BE

            elif self.direction == "SELL":
                if current_price <= self.entry_price - breakeven_trigger_diff:
                    self.stop_loss = min(self.stop_loss, self.entry_price)

        # Trailing stop logic
        if (
            self.config.get("trail_start") is not None
            and self.config.get("trail_distance") is not None
        ):
            # Diff away from entry price where we start trailing
            trail_start_diff = self.config["trail_start"]

            # Distance away from current price where we will move SL
            trail_distance = self.config["trail_distance"]

            if self.direction == "BUY":
                # Price where we will move SL to "trail_distance" away
                trail_trigger_price = self.entry_price + trail_start_diff

                # If we cross trigger price, update SL
                if current_price >= trail_trigger_price:
                    new_sl = current_price - trail_distance

                    # If this is the first SL update, or if the new SL is higher than the previous one
                    if not self.trailing_steps or new_sl > max(self.trailing_steps):
                        self.trailing_steps.append(new_sl)
                        self.stop_loss = max(self.stop_loss, new_sl)
            # Opposite logic for SELL
            elif self.direction == "SELL":
                if current_price <= self.entry_price - trail_start_diff:
                    new_sl = current_price + trail_distance
                    if not self.trailing_steps or new_sl < min(self.trailing_steps):
                        self.trailing_steps.append(new_sl)
                        self.stop_loss = min(self.stop_loss, new_sl)

        return self.stop_loss

    def check_exit(self, current_price: float) -> tuple[str, float] | None:
        """
        Checks whether SL or TP was hit and returns the appropriate exit info.

        Returns:
            Tuple of ('STOP_LOSS' or 'TAKE_PROFIT', price_hit) or None
        """
        if self.direction == "BUY":
            if self.stop_loss is not None and current_price <= self.stop_loss:
                self.hit_stop_loss = True
                return ("STOP_LOSS", self.stop_loss)
            if self.take_profit is not None and current_price >= self.take_profit:
                self.hit_take_profit = True
                return ("TAKE_PROFIT", self.take_profit)

        elif self.direction == "SELL":
            if self.stop_loss is not None and current_price >= self.stop_loss:
                self.hit_stop_loss = True
                return ("STOP_LOSS", self.stop_loss)
            if self.take_profit is not None and current_price <= self.take_profit:
                self.hit_take_profit = True
                return ("TAKE_PROFIT", self.take_profit)

        return None

    def update(self, current_price: float) -> tuple[str, float, str | None]:
        """
        Updates stop loss and checks for exit conditions.

        Returns:
            (exit_type ('STOP_LOSS' | 'TAKE_PROFIT' | None), exit_price (float | None), event (str | None))
        """
        exit_result = self.check_exit(current_price)
        if exit_result:
            exit_type, exit_price = exit_result
            return exit_type, exit_price, None

        event = None
        old_sl = self.stop_loss
        self.update_stop_loss(current_price)
        if self.stop_loss != old_sl:
            if self.stop_loss == self.entry_price:
                event = "Breakeven Hit"
            else:
                event = "Trailing SL Updated"

        return None, None, event


class PositionCalculator:
    """
    Class to calculate position parameters such as stop loss, pip values, atr (NNFX)
    """

    def calculate_pip_size(forex_pair: str):
        """
        Calculates the pip size based on the provided forex pair. (NNFX)
        """
        pip_size = 0.01 if "JPY" in forex_pair else 0.0001

        return pip_size

    def calculate_atr_pips(forex_pair: str, atr: float):
        """
        Calculates the ATR in pips based on the provided ATR value and forex pair. (NNFX)
        """
        pip_size = PositionCalculator.calculate_pip_size(forex_pair)

        return atr / pip_size

    def calculate_target_pip_value(
        balance: int, risk_per_trade: float, stop_loss_in_pips
    ):
        """
        Calculates the target pip value (dollars ($) per pip) based on balance,
        risk per trade, and stop loss in pips. (NNFX)
        """
        risk = balance * risk_per_trade

        return round(risk / stop_loss_in_pips, 2)

    def calculate_current_pip_value(
        forex_pair: str,
        units: float,
        exchange_rate: float,
        quote_to_usd_rate: float = None,
    ) -> float:
        """
        Calculate pip value in USD for any currency pair.

        Args:
            forex_pair (str): Currency pair (e.g. EUR/USD)
            units (float): Trade volume (e.g. 10,000 for mini lot, 100,000 for standard)
            exchange_rate (float): Exchange rate of the currency pair (current price)
            quote_to_usd_rate (float): Exchange rate of quote currency to USD (only needed for cross pairs such as EUR/GBP)

        Returns:
            float: Pip value in USD
        """
        pip_size = PositionCalculator.calculate_pip_size(forex_pair)
        base_currency, quote_currency = forex_pair[:3], forex_pair[3:]

        # USD is the quote currency (e.g. EUR/USD)
        if quote_currency == "USD":
            pip_value = units * pip_size

        # USD is the base currency (e.g. USD/CAD)
        elif base_currency == "USD":
            pip_value = units * pip_size / exchange_rate

        # Cross-currency pair (e.g. EUR/GBP), convert pip value to USD
        else:
            pip_value_in_quote = units * pip_size
            pip_value = pip_value_in_quote * quote_to_usd_rate

        return round(pip_value, 2)

    def calculate_stop_loss_pips(
        forex_pair: str, atr: float, atr_multiplier: float = 1.5
    ):
        """
        Calculates the stop loss in pips based on the provided ATR value and multiplier. (NNFX)
        """
        return PositionCalculator.calculate_atr_pips(forex_pair, atr) * atr_multiplier

    def calculate_stop_loss_price(direction: str, entry_price: float, atr: float):
        """
        Calculates the stop loss to be ATR * multiplier AWAY from the entry price. (NNFX)
        """
        # Calculate stop loss distance in pips
        stop_loss_in_pips = PositionCalculator.calculate_stop_loss_pips(atr)

        if direction == "BUY":
            stop_loss_in_pips = -stop_loss_in_pips
        elif direction == "SELL":
            stop_loss_in_pips = stop_loss_in_pips

        # Remove decimal from entry price
        str_price = str(entry_price)  # Convert to string
        decimal_position = str_price.find(".")  # Get decimal position

        # No decimal found, treat as an integer adjustment
        if decimal_position == -1:
            return entry_price + stop_loss_in_pips

        raw_digits = int(
            str_price.replace(".", "")
        )  # Remove the decimal from entry price
        adjusted_digits = raw_digits + stop_loss_in_pips  # Apply the ATR pip adjustment

        # Restore decimal at the original position, removing decimal added in str() conversion
        adjusted_str = str(adjusted_digits).replace(".", "")
        stop_loss_price = (
            adjusted_str[:decimal_position] + "." + adjusted_str[decimal_position:]
        )

        return float(stop_loss_price)

    def calculate_trade_units(
        forex_pair: str,
        balance: int,
        risk_per_trade: float,
        entry_price: float,
        atr: float,
        quote_to_usd_rate: float = None,
    ):
        """
        Calculates the number of trade units based on the target pip value. (NNFX)
        """
        # Calculate current pip value per lot
        dollars_per_pip_per_lot = PositionCalculator.calculate_current_pip_value(
            forex_pair, 100_000, entry_price, quote_to_usd_rate
        )

        # Calculate stop loss in pips
        stop_loss_in_pips = PositionCalculator.calculate_stop_loss_pips(forex_pair, atr)

        # Calculate target pip value
        target_pip_value = PositionCalculator.calculate_target_pip_value(
            balance, risk_per_trade, stop_loss_in_pips
        )

        # Calculate trade units
        units = target_pip_value * 100_000 / dollars_per_pip_per_lot

        return round(units)

    def calculate_required_margin(
        forex_pair: str,
        units: int,
        leverage: int,
        price: float,
        quote_to_usd_rate: float = None,
    ) -> float:
        """
        Calculates required margin in USD for a forex trade.

        Args:
            forex_pair (str): e.g. 'EURUSD', 'EURAUD', etc.
            units (int): Trade size in base currency units (e.g., 250000)
            leverage (int): Leverage factor (e.g., 500)
            price (float): Exchange rate of the pair (quote currency per base currency)
            quote_to_usd_rate (float, optional): Conversion rate from quote currency to USD.
                                                Required if USD is not the quote currency.

        Returns:
            float: Required margin in USD
        """
        # Extract the quote currency (last 3 letters)
        quote_currency = forex_pair[-3:]

        # Margin in quote currency
        margin_in_quote = (units * price) / leverage

        # If quote currency is USD, no conversion needed
        if quote_currency == "USD":
            return round(margin_in_quote, 2)

        # Otherwise, convert to USD (must have quote_to_usd_rate)
        if quote_to_usd_rate is None:
            raise ValueError(
                f"quote_to_usd_rate is required for non-USD quote currency ({quote_currency})"
            )

        margin_in_usd = margin_in_quote * quote_to_usd_rate
        return round(margin_in_usd, 2)

    def calculate_profit_from_pips(
        forex_pair: str,
        pip_change: float,
        units: float,
        exit_price: float,
        quote_to_usd_rate: float = None,
    ) -> float:
        """
        Calculates profit in USD from pip change and lot size using exit price for accurate pip value.

        Args:
            forex_pair (str): Forex pair
            pip_change (float): Number of pips gained/lost
            units (float): Trade volume
            exit_price (float): Exchange rate at exit
            quote_to_usd_rate (float): Optional, needed for cross pairs

        Returns:
            float: Profit in USD
        """
        pip_value = PositionCalculator.calculate_current_pip_value(
            forex_pair, units, exit_price, quote_to_usd_rate
        )

        return round(pip_change * pip_value, 2)

    def calculate_pip_change(
        forex_pair: str, entry_price: float, exit_price: float, direction: str
    ) -> float:
        """
        Calculates pip change between entry and exit price.

        Args:
            forex_pair (str): Forex pair
            entry_price (float): Entry price of the trade
            exit_price (float): Exit price of the trade
            direction (str): 'BUY' or 'SELL'

        Returns:
            float: Number of pips gained or lost (can be positive or negative)
        """
        pip_size = PositionCalculator.calculate_pip_size(forex_pair)

        price_diff = (
            exit_price - entry_price if direction == "BUY" else entry_price - exit_price
        )

        pip_change = price_diff / pip_size

        return round(pip_change, 1)


class BaseStrategy:
    NAME = None
    DESCRIPTION = None
    PARAMETER_SETTINGS = {}
    FOREX_PAIR = None
    CONFIG_ID = None

    def __init__(self, forex_pair: str, parameters: dict = None):
        self.FOREX_PAIR = forex_pair
        self.PARAMETER_SETTINGS = parameters

    def prepare_data(self, full_data: pd.DataFrame):
        """
        Precomputes and stores indicator outputs for the full dataset.
        Call this once before backtest loop.
        """
        self.data_with_indicators = self._calculate_indicators(full_data)

    def generate_trade_plan(
        self,
        data: pd.DataFrame,
        current_position: int,
        balance: float,
        leverage: int,
        quote_to_usd_rate: float,
    ) -> list[TradePlan]:
        return []

    def _generate_signals(
        self, data: pd.DataFrame, current_position: int, debug: bool = False
    ) -> list[str] | tuple[list[str], dict]:
        raise NotImplementedError(
            "Please implement the generate_signal method in your strategy subclass."
        )

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all NNFX indicators (ATR, Baseline, C1, C2, Volume, Exit)
        and returns a DataFrame with their outputs attached.
        """
        raise NotImplementedError(
            "Please implement the calculate_indicators method in your strategy subclass."
        )

    # Extract first series if indicator function returns a DataFrame
    def _convert_np_to_pd(
        self, output: pd.DataFrame | pd.Series | tuple | np.ndarray
    ) -> pd.DataFrame | pd.Series:
        if isinstance(output, pd.DataFrame) or isinstance(output, pd.Series):
            return output
        elif isinstance(output, np.ndarray):
            return pd.Series(output)
        elif isinstance(output, tuple):
            # If it's a tuple of Series or arrays, build a DataFrame with auto-named columns
            converted = []

            for i, item in enumerate(output):
                if isinstance(item, pd.Series):
                    converted.append(item.reset_index(drop=True))
                elif isinstance(item, np.ndarray):
                    converted.append(pd.Series(item))
                else:
                    raise TypeError(
                        f"Unsupported type in tuple at index {i}: {type(item)}"
                    )

            return pd.concat(converted, axis=1)
        else:
            raise TypeError(f"Unsupported type for conversion: {type(output)}")

    def _reconstruct_df(
        self, data: pd.DataFrame, prefix: str
    ) -> pd.DataFrame | pd.Series:
        """Reconstruct a DataFrame or Series from columns that start with a given prefix

        Args:
            data (pd.DataFrame): The DataFrame to reconstruct from
            prefix (str): The prefix to search for

        Raises:
            KeyError: No columns found for the given prefix

        Returns:
            pd.DataFrame | pd.Series: The reconstructed DataFrame, or Series if only one column
        """
        cols = [
            col for col in data.columns if col == prefix or col.startswith(f"{prefix}_")
        ]
        if not cols:
            raise KeyError(f"No columns found for prefix '{prefix}'")
        if len(cols) == 1 and cols[0] == prefix:
            return data[cols[0]]
        return data[cols]

    def _get_slice(self, end_index: int) -> pd.DataFrame:
        """
        Returns a slice of the precomputed dataset up to a given index.
        """
        if self.data_with_indicators is None:
            raise ValueError("Indicators not precomputed. Call `prepare_data()` first.")
        return self.data_with_indicators.iloc[: end_index + 1]

    def _get_indicator_signals(
        self,
        signal_fn: Callable,
        indicator_fn_output: pd.DataFrame | pd.Series,
        close_series: pd.Series = None,
    ) -> list[str]:
        """Interprets and retrieves indicator signals

        Args:
            signal_fn (Callable): The signal function
            indicator_fn_output (pd.DataFrame | pd.Series): The output of the indicator function
            close_series (pd.Series, optional): _description_. Defaults to None.

        Returns:
            list[str]: The list of signals
        """
        sig = inspect.signature(signal_fn)
        params = list(sig.parameters.values())

        if len(params) == 2 and close_series is not None:
            second_param = params[1]
            annotation = second_param.annotation

            # If the second parameter is annotated and it's a DataFrame/Series, pass current_row
            if annotation in [pd.Series, pd.DataFrame] or annotation is inspect._empty:
                # Runtime check as backup: only pass current_row if it's a Series and fn doesn't crash
                try:
                    return signal_fn(indicator_fn_output, close_series)
                except Exception:
                    return signal_fn(indicator_fn_output)

        # One-argument or incompatible second parameter
        return signal_fn(indicator_fn_output)

    def __repr__(self):
        return f"{self.NAME} ({self.PARAMETER_SETTINGS})"


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
    ):

        self.atr = atr
        self.baseline = baseline
        self.c1 = c1
        self.c2 = c2
        self.volume = volume
        self.exit = exit_indicator

        self.parameters = {
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

        super().__init__(forex_pair=forex_pair, parameters=self.parameters)

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all NNFX indicators (ATR, Baseline, C1, C2, Volume, Exit)
        and returns a DataFrame with their outputs attached.
        """
        data = data.copy()

        def safe_apply(name: str, config: IndicatorConfig):
            raw_output = config["function"](data, **config["parameters"])
            output_df_or_series = self._convert_np_to_pd(raw_output)

            if isinstance(output_df_or_series, pd.DataFrame):
                renamed = output_df_or_series.add_prefix(f"{name}_")
                return renamed
            else:
                return pd.DataFrame({name: output_df_or_series})

        # Collect all indicator DataFrames
        indicator_dfs = [
            safe_apply("ATR", self.atr),
            safe_apply("Baseline", self.baseline),
            safe_apply("C1", self.c1),
            safe_apply("C2", self.c2),
            safe_apply("VolumeIndicator", self.volume),
            safe_apply("Exit", self.exit),
        ]

        # Concatenate once to avoid fragmentation
        indicator_data = pd.concat(indicator_dfs, axis=1)
        data = pd.concat(
            [data.reset_index(drop=True), indicator_data.reset_index(drop=True)], axis=1
        )

        return data

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
        current_row = data.iloc[-1]
        previous_row = data.iloc[-2]

        atr_series = data["ATR"]
        baseline_series = data["Baseline"]
        c1_df_or_series = (
            self._reconstruct_df(data, "C1")
            # if "C1" in data
            # else data[[col for col in data if col.startswith("C1_")][0]]
        )
        c2_df_or_series = self._reconstruct_df(data, "C2")
        volume_df_or_series = self._reconstruct_df(data, "VolumeIndicator")
        exit_df_or_series = self._reconstruct_df(data, "Exit")

        # Remove tags from column names in DataFrames for passing to signal functions
        if isinstance(atr_series, pd.DataFrame):
            atr_series.columns = [col.replace("ATR_", "") for col in atr_series.columns]
        if isinstance(baseline_series, pd.DataFrame):
            baseline_series.columns = [
                col.replace("Baseline_", "") for col in baseline_series.columns
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

        # print(atr_series.head())
        # print(baseline_series.head())
        # print(c1_df_or_series.head())
        # print(c2_df_or_series.head())
        # print(volume_df_or_series.head())
        # print(exit_df_or_series.head())

        close_series = data["Close"]

        # Evaluate signal functions
        atr_signals = self._get_indicator_signals(
            self.atr["signal_function"], atr_series, close_series
        )
        baseline_signals = self._get_indicator_signals(
            self.baseline["signal_function"], baseline_series, close_series
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
        prev_close_series = close_series.iloc[:-1]
        baseline_signals_prev = self._get_indicator_signals(
            self.baseline["signal_function"],
            baseline_series.iloc[:-1],
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
        atr_value = atr_series.iloc[-1]
        prev_atr_value = atr_series.iloc[-2]
        baseline_value = baseline_series.iloc[-1]

        # Check conditions
        price_within_1x_atr_of_baseline = (
            abs(current_row["Close"] - baseline_value) < 1 * atr_value
        )
        prev_price_within_1x_atr_of_baseline = (
            abs(previous_row["Close"] - baseline_value) < 1 * prev_atr_value
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
        flattened_signals = [sig for sublist in recent_c1_signals for sig in sublist]

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
                # Get data since the last valid signal
                data_since_signal = data.iloc[self.last_valid_entry_index + 1 :]

                # Has price crossed BELOW the baseline since the last valid signal?
                below_baseline_since_signal = (
                    data_since_signal["Close"] < data_since_signal["Baseline"]
                ).any()

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
                data_since_signal = data.iloc[self.last_valid_entry_index + 1 :]

                # Has price crossed ABOVE the baseline since the last valid signal?
                above_baseline_since_signal = (
                    data_since_signal["Close"] > data_since_signal["Baseline"]
                ).any()

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
                or c1_bearish_signal
                or c2_bearish_signal
            ):
                signals.append(EXIT_LONG)
                signal_source = "Exit"

        elif current_position == -1:
            if (
                exit_bullish_signal
                or baseline_bullish_signal
                or c1_bullish_signal
                or c2_bullish_signal
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
            atr = row["ATR"]
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
                    strategy="NNFX",
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
                strategy="NNFX",
                direction=direction,
                entry_price=row["Close"],
                stop_loss=None,
                tag="EXIT",
                source=signal_source,
            )

            trade_plans.append(exit_plan)

        return trade_plans
