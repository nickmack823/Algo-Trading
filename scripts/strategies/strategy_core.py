import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Type, runtime_checkable

import pandas as pd


# --- Strategy registry & factory (ADD) --------------------------------------
@dataclass(frozen=True)
class StrategyMeta:
    key: str
    cls: Type["BaseStrategy"]
    required_kwargs: List[str]
    description: str = ""


STRATEGY_REGISTRY: Dict[str, StrategyMeta] = {}


def register_strategy(
    key: str, required_kwargs: List[str] | None = None, description: str = ""
):
    """Class decorator to register a strategy so main.py can instantiate by name."""

    def _wrap(cls: Type["BaseStrategy"]):
        STRATEGY_REGISTRY[key] = StrategyMeta(
            key=key,
            cls=cls,
            required_kwargs=required_kwargs or [],
            description=description or getattr(cls, "DESCRIPTION", ""),
        )
        return cls

    return _wrap


def get_registered_strategies() -> Dict[str, StrategyMeta]:
    """Return a dictionary of all registered strategies."""
    return dict(STRATEGY_REGISTRY)


def create_strategy_from_kwargs(key: str, **kwargs) -> "BaseStrategy":
    """Create a strategy instance by name (key) and kwargs."""
    meta = STRATEGY_REGISTRY.get(key)
    if meta is None:
        raise KeyError(
            f"Strategy '{key}' is not registered. Known: {list(STRATEGY_REGISTRY)}"
        )
    missing = [k for k in meta.required_kwargs if k not in kwargs]
    if missing:
        raise TypeError(f"Strategy '{key}' missing required args: {missing}")
    return meta.cls(**kwargs)


# ---------------------------------------------------------------------------


@runtime_checkable
class StrategyProtocol(Protocol):
    NAME: str
    DESCRIPTION: str
    PARAMETER_SETTINGS: dict
    FOREX_PAIR: str
    TIMEFRAME: str

    def prepare_data(
        self, historical_data: pd.DataFrame, use_cache: bool = True
    ) -> None: ...
    def get_cache_jobs(self) -> list[dict]: ...
    def generate_trade_plan(
        self,
        current_index: int,
        current_position: int,
        balance: float,
        quote_to_usd_rate: Optional[float],
    ) -> List["TradePlan"]: ...


class StrategySignal:
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_SHORT = "EXIT_SHORT"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class TradePlan:
    strategy: str | None  # default None; fill with self.NAME when emitting
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

        # Before updating stop loss, store old stop loss
        old_sl = self.stop_loss
        self.update_stop_loss(current_price)

        if self.stop_loss != old_sl:
            # Special check: Breakeven triggers ONLY if stop moved exactly to entry price AND was BELOW entry price before
            if (
                old_sl < self.entry_price
                and self.stop_loss == self.entry_price
                and self.direction == "BUY"
            ) or (
                old_sl > self.entry_price
                and self.stop_loss == self.entry_price
                and self.direction == "SELL"
            ):
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

        return pip_value

    def calculate_stop_loss_pips(
        forex_pair: str, atr: float, atr_multiplier: float = 1.5
    ):
        """
        Calculates the stop loss in pips based on the provided ATR value and multiplier. (NNFX)
        """
        return PositionCalculator.calculate_atr_pips(forex_pair, atr) * atr_multiplier

    def calculate_stop_loss_price(
        forex_pair: str, direction: str, entry_price: float, atr: float
    ):
        """
        Calculates the stop loss to be ATR * multiplier AWAY from the entry price. (NNFX)
        """
        # Calculate stop loss distance in pips
        stop_loss_in_pips = PositionCalculator.calculate_stop_loss_pips(forex_pair, atr)

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

        Args:
            forex_pair (str): e.g. 'EURUSD', 'EURAUD', etc. (no "/")
            balance (int): Available balance
            risk_per_trade (float): Risk per trade
            entry_price (float): Entry price
            atr (float): Average True Range
            quote_to_usd_rate (float): Optional, needed for cross pairs
        """
        assert "/" not in forex_pair, "forex_pair must not contain '/'"

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
    TIMEFRAME = None
    CONFIG_ID = None

    def __init__(self, forex_pair: str, parameters: dict = None, timeframe: str = None):
        self.FOREX_PAIR = forex_pair.replace("/", "")
        self.PARAMETER_SETTINGS = parameters
        self.TIMEFRAME = timeframe

    # def prepare_data(self, full_data: pd.DataFrame):
    #     """
    #     Precomputes and stores indicator outputs for the full dataset.
    #     Call this once before backtest loop.
    #     """
    #     self.data_with_indicators = self._calculate_indicators(full_data)

    def generate_trade_plan(
        self,
        current_index: int,
        current_position: int,
        balance: float,
        quote_to_usd_rate: float,
    ) -> list[TradePlan]:
        """
        Default no-op. Strategies should override this.
        Tip: use self._get_slice(current_index) to access precomputed data
        prepared in prepare_data(...).
        """
        return []

    def prepare_data(self, historical_data: pd.DataFrame, use_cache: bool = True):
        """
        Default: keep raw data so subclasses can use _get_slice(...)
        Override in your strategy if you compute indicators.
        """
        self.data_with_indicators = historical_data

    def get_cache_jobs(self) -> list[dict]:
        """
        Default: no indicator-cache work to persist.
        Override if your strategy stores indicator outputs (like NNFX).
        """
        return []

    def _generate_signals(
        self, data: pd.DataFrame, current_position: int, debug: bool = False
    ) -> list[str] | tuple[list[str], dict]:
        raise NotImplementedError(
            "Please implement the generate_signal method in your strategy subclass."
        )

    def _calculate_or_retrieve_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all NNFX indicators (ATR, Baseline, C1, C2, Volume, Exit)
        and returns a DataFrame with their outputs attached.
        """
        raise NotImplementedError(
            "Please implement the calculate_indicators method in your strategy subclass."
        )

    # Extract first series if indicator function returns a DataFrame
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
        cols = [col for col in data.columns if col.startswith(f"{prefix}_")]
        if not cols:
            raise KeyError(f"No columns found for prefix '{prefix}'")
        # If only one column, return it as a Series
        if len(cols) == 1 and cols[0].startswith(f"{prefix}_"):
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

        if len(params) == 2 and isinstance(close_series, pd.Series):
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


def load_strategy_from_dict(d: dict) -> BaseStrategy:
    """
    Expected dict shape (current NNFX):
      {
        "strategy_name": "NNFX",            # optional; default "NNFX"
        "pair": "EURUSD",
        "timeframe": "4_hour",
        "parameters": {
           "ATR": <IndicatorConfig>,
           "Baseline": <IndicatorConfig>,
           "C1": <IndicatorConfig>,
           "C2": <IndicatorConfig>,
           "Volume": <IndicatorConfig>,
           "Exit": <IndicatorConfig>,
        }
      }
    """
    name = d.get("strategy_name", None)
    pair = d["pair"]
    timeframe = d["timeframe"]

    params = d.get("parameters")
    if params is None:
        # Reconstruct params dict from your current row shape if needed
        # (Your existing code that maps raw fields to IndicatorConfig goes here)
        raise ValueError("Expected 'parameters' dict in strategy row.")

    return create_strategy_from_kwargs(name, params, pair, timeframe)
