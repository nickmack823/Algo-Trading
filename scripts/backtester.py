import math
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from scripts import config
from scripts.data.sql import BacktestSQLHelper, HistoricalDataSQLHelper
from scripts.strategies import strategy_core
from scripts.trading_hours import build_entry_allowed_mask


class Trade:
    """
    A class to store details of a trade, including entry, exit, PnL, and commissions.
    """

    def __init__(
        self,
        plan: strategy_core.TradePlan,
        forex_pair: str,
        entry_timestamp,
        entry_price: float,
        balance: float,
        leverage: int,
        margin_required: float,
        pip_value: float,
    ):
        self.plan = plan
        self.forex_pair = forex_pair
        self.entry_timestamp = entry_timestamp  # Time of trade entry
        self.entry_price = entry_price
        self.balance_before_trade = balance  # Account balance before the trade
        self.balance_after_trade = balance  # Account balance after the trade
        self.leverage = leverage  # Leverage factor
        self.margin_required = margin_required  # Margin required for the trade
        self.pip_value = pip_value  # Pip value of the trade

        self.position_events = []  # Like breakeven triggers, trailing SL triggers

        self.direction = plan.direction  # 'BUY' or 'SELL'
        self.units = plan.units  # Size of the trade in units (100,000 units/lot)

        self.num_decimals = 3 if "JPY" in self.forex_pair else 5
        self.entry_price = round(self.entry_price, self.num_decimals)

        self.lot_size = round(self.units / 100_000, 2)  # Size of the trade in lots

        self.exit_timestamp = None  # Time of trade exit
        self.exit_price = None  # Price at exit
        self.pnl = None  # Profit or Loss from the trade
        self.commission = None  # Commission paid for the trade
        self.net_pips = None  # Number of pips gained or lost

        # Performance metrics
        self.duration = None
        self.return_pct = None
        self.is_win = None

    def close_trade(
        self,
        exit_timestamp,
        exit_price: float,
        commission_per_lot: float,
        quote_to_usd_exit_rate: float = None,
        reason: str = None,
    ):
        if self.exit_price is not None:
            return  # Prevent closing a trade twice

        self.exit_price = exit_price

        self.net_pips = strategy_core.PositionCalculator.calculate_pip_change(
            self.forex_pair, self.entry_price, self.exit_price, self.direction
        )

        # Calculate PnL based on trade direction
        self.pnl = strategy_core.PositionCalculator.calculate_profit_from_pips(
            self.forex_pair,
            self.net_pips,
            self.units,
            self.exit_price,
            quote_to_usd_exit_rate,
        )

        # Calculate commission based on lot size
        self.commission = round(self.lot_size * commission_per_lot, 2)

        # Update balance after accounting for PnL and commission
        self.balance_after_trade += self.pnl - self.commission
        self.balance_after_trade = round(self.balance_after_trade, 2)
        self.exit_timestamp = exit_timestamp

        # Round exit price
        self.exit_price = round(self.exit_price, self.num_decimals)

        # Mark reason for exiting (STOP_LOSS, TAKE_PROFIT, etc.)
        self.exit_reason = reason

        # Set performance metrics
        entry_time = pd.to_datetime(self.entry_timestamp)
        exit_time = pd.to_datetime(self.exit_timestamp)
        self.duration = int(
            (exit_time - entry_time).total_seconds() / 60
        )  # duration in minutes
        self.return_pct = (
            round((self.pnl / self.balance_before_trade) * 100, 2)
            if self.balance_before_trade != 0
            else None
        )

    def __repr__(self):
        return (
            f"{self.entry_timestamp}: {self.direction} @ {self.entry_price}, "
            f"Lot Size: {self.lot_size}, Exit: {self.exit_price}, "
            f"PnL: ${self.pnl}, Pip Change: {self.net_pips}, Commission: {self.commission}, "
            f"Return %: {self.return_pct}, Duration (min): {self.duration}, "
            f"Balance: ${self.balance_after_trade}, Leverage: {self.leverage}, Margin Required: ${self.margin_required}, Pip Value: {self.pip_value}"
        )


class Backtester:
    def __init__(
        self,
        strategy: strategy_core.BaseStrategy,
        forex_pair: str,  # e.g. "EURUSD" with no slashes
        timeframe: str,
        data: pd.DataFrame,
        initial_balance=10_000,
        commission_per_lot=6,
        slippage=0.0002,
        leverage=50,
        metrics_start_date: str | None = None,
        metrics_end_date: str | None = None,
        intrabar_mode: str = "hybrid_ohlc",
        intrabar_timeframe: str | None = None,
        intrabar_data: pd.DataFrame | None = None,
        session_config: dict | None = None,
    ):
        """
        Initializes the backtester with historical data and trading parameters.
        """
        self.strategy = strategy  # Trading strategy
        self.forex_pair = forex_pair.replace("/", "")  # Forex pair being traded
        self.base_currency, self.quote_currency = (
            self.forex_pair[:3],
            self.forex_pair[3:],
        )

        self.timeframe = timeframe

        self.data = data
        self.data_start_date, self.data_end_date = (
            self.data["Timestamp"].min(),
            self.data["Timestamp"].max(),
        )
        self.metrics_start_date = metrics_start_date
        self.metrics_end_date = metrics_end_date
        self.intrabar_mode = intrabar_mode
        self.intrabar_timeframe = intrabar_timeframe

        self.initial_balance = initial_balance  # Starting account balance
        self.balance = initial_balance  # Current account balance
        self.commission_per_lot = commission_per_lot  # Commission charged per lot (USD)
        self.slippage = slippage  # Slippage factor
        self.leverage = leverage  # Leverage factor

        # Keep track of margin remaining
        self.margin_remaining = initial_balance

        # SQL Helpers
        self.backtest_sqlhelper = BacktestSQLHelper()

        # self.forex_pair_id = self.backtest_sqlhelper.get_forex_pair_id(
        #     f"{self.base_currency}/{self.quote_currency}"
        # )
        # self.strategy_config_id = self.backtest_sqlhelper.upsert_strategy_configuration(
        #     self.strategy.NAME,
        #     self.strategy.DESCRIPTION,
        #     self.strategy.PARAMETER_SETTINGS,
        # )

        # # Precompute indicators (moved outside for multiprocessing)
        # self.strategy.prepare_data(self.data)

        # Set a start index for when we start trading on the data (to create a semblance of historical data)
        self.trading_start_index = 100

        # Quote->USD conversion series (used for non-USD quote pairs).
        self.quote_to_usd_data = None
        self._quote_to_usd_parent_timestamps: np.ndarray | None = None
        self._quote_to_usd_parent_values: np.ndarray | None = None
        self._quote_to_usd_intrabar_timestamps: np.ndarray | None = None
        self._quote_to_usd_intrabar_values: np.ndarray | None = None

        if self.quote_currency != "USD":
            self._prepare_parent_quote_to_usd_data()

        self.all_trades: list[Trade] = []  # List of all executed trades
        self.open_trades: list[Trade] = []  # Stores open trades
        self.position = 0  # Current position: 1 for long, -1 for short, 0 for none
        self.max_drawdown = 0  # Maximum observed drawdown
        self.max_drawdown_pct = 0  # Maximum observed drawdown as a percentage
        self.peak_balance = initial_balance  # Highest account balance observed
        self.equity_samples: list[tuple[pd.Timestamp, float]] = []
        self._parent_bar_delta = self._timeframe_to_timedelta(self.timeframe)
        self.session_config = session_config

        # Lower-timeframe arrays used only when intrabar_mode == "lower_timeframe".
        self._intrabar_timestamps: np.ndarray | None = None
        self._intrabar_highs: np.ndarray | None = None
        self._intrabar_lows: np.ndarray | None = None
        self._intrabar_closes: np.ndarray | None = None
        self._prepare_intrabar_data(intrabar_data)
        self._entry_allowed_mask = build_entry_allowed_mask(
            self.data["Timestamp"], self.session_config, timeframe=self.timeframe
        )

        if self.quote_currency != "USD":
            self._prepare_intrabar_quote_to_usd_data()

        # Close SQL connections
        self.backtest_sqlhelper.close_connection()
        self.backtest_sqlhelper = None

    @staticmethod
    def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
        try:
            amount_str, unit = timeframe.split("_", 1)
            amount = int(amount_str)
        except Exception:
            return pd.Timedelta(minutes=1)

        if unit == "minute":
            return pd.Timedelta(minutes=amount)
        if unit == "hour":
            return pd.Timedelta(hours=amount)
        if unit == "day":
            return pd.Timedelta(days=amount)
        return pd.Timedelta(minutes=1)

    @staticmethod
    def _sanitize_quote_df(quote_df: pd.DataFrame | None) -> pd.DataFrame:
        if quote_df is None or quote_df.empty:
            return pd.DataFrame(columns=["Timestamp", "Close"])

        if "Timestamp" not in quote_df.columns or "Close" not in quote_df.columns:
            return pd.DataFrame(columns=["Timestamp", "Close"])

        out = quote_df.loc[:, ["Timestamp", "Close"]].copy()
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
        out = out.dropna(subset=["Timestamp", "Close"])
        out = out.sort_values("Timestamp")
        out = out.drop_duplicates(subset=["Timestamp"], keep="last")
        out = out.reset_index(drop=True)
        return out

    def _prepare_parent_quote_to_usd_data(self) -> None:
        secondary_data_sqlhelper = None
        try:
            secondary_data_sqlhelper = HistoricalDataSQLHelper(
                f"{config.DATA_FOLDER}/{self.quote_currency}USD.db"
            )
            raw_quote_df = secondary_data_sqlhelper.get_historical_data(table=self.timeframe)
            quote_df = self._sanitize_quote_df(raw_quote_df)

            if quote_df.empty:
                self.quote_to_usd_data = None
                return

            aligned = quote_df.set_index("Timestamp")
            data_timestamps = pd.to_datetime(self.data["Timestamp"], errors="coerce")
            data_timestamps = data_timestamps.dropna().drop_duplicates().sort_values()

            all_timestamps = aligned.index.union(data_timestamps)
            aligned = aligned.reindex(all_timestamps).ffill()
            aligned = aligned.reindex(data_timestamps)
            aligned = aligned.dropna(subset=["Close"]).reset_index()

            if aligned.empty:
                self.quote_to_usd_data = None
                return

            self.quote_to_usd_data = aligned
            self._quote_to_usd_parent_timestamps = aligned["Timestamp"].to_numpy(
                dtype="datetime64[ns]"
            )
            self._quote_to_usd_parent_values = aligned["Close"].to_numpy(dtype=float)
        except Exception as e:
            logging.warning(
                "Failed loading parent quote->USD data for %s (%s): %s",
                self.forex_pair,
                self.timeframe,
                e,
            )
            self.quote_to_usd_data = None
        finally:
            if secondary_data_sqlhelper is not None:
                secondary_data_sqlhelper.close_connection()

    def _prepare_intrabar_quote_to_usd_data(self) -> None:
        if self.intrabar_mode != "lower_timeframe":
            return
        if not self.intrabar_timeframe or self.intrabar_timeframe == self.timeframe:
            return

        secondary_data_sqlhelper = None
        try:
            from_date = pd.to_datetime(self.data_start_date).strftime("%Y-%m-%d")
            to_date = pd.to_datetime(self.data_end_date).strftime("%Y-%m-%d")

            secondary_data_sqlhelper = HistoricalDataSQLHelper(
                f"{config.DATA_FOLDER}/{self.quote_currency}USD.db"
            )
            raw_quote_df = secondary_data_sqlhelper.get_historical_data(
                table=self.intrabar_timeframe, from_date=from_date, to_date=to_date
            )
            quote_df = self._sanitize_quote_df(raw_quote_df)
            if quote_df.empty:
                return

            self._quote_to_usd_intrabar_timestamps = quote_df["Timestamp"].to_numpy(
                dtype="datetime64[ns]"
            )
            self._quote_to_usd_intrabar_values = quote_df["Close"].to_numpy(dtype=float)
        except Exception as e:
            logging.warning(
                "Failed loading intrabar quote->USD data for %s (%s -> %s): %s",
                self.forex_pair,
                self.timeframe,
                self.intrabar_timeframe,
                e,
            )
        finally:
            if secondary_data_sqlhelper is not None:
                secondary_data_sqlhelper.close_connection()

    @staticmethod
    def _lookup_rate_asof(
        timestamps: np.ndarray | None,
        values: np.ndarray | None,
        ts,
        fallback: float | None,
    ) -> float | None:
        if timestamps is None or values is None or len(timestamps) == 0:
            return fallback

        try:
            ts_np = np.datetime64(pd.to_datetime(ts))
        except Exception:
            return fallback

        idx = int(np.searchsorted(timestamps, ts_np, side="right") - 1)
        if idx < 0 or idx >= len(values):
            return fallback

        rate = values[idx]
        if not np.isfinite(rate):
            return fallback
        return float(rate)

    def _resolve_quote_to_usd_rate_for_exit(
        self, exit_timestamp, parent_fallback_rate: float | None
    ) -> float | None:
        if self.quote_currency == "USD":
            return None

        rate = parent_fallback_rate
        if self.intrabar_mode == "lower_timeframe":
            rate = self._lookup_rate_asof(
                self._quote_to_usd_intrabar_timestamps,
                self._quote_to_usd_intrabar_values,
                exit_timestamp,
                fallback=rate,
            )

        rate = self._lookup_rate_asof(
            self._quote_to_usd_parent_timestamps,
            self._quote_to_usd_parent_values,
            exit_timestamp,
            fallback=rate,
        )
        return rate

    def _prepare_intrabar_data(self, intrabar_data: pd.DataFrame | None) -> None:
        allowed_modes = {"hybrid_ohlc", "lower_timeframe"}
        if self.intrabar_mode not in allowed_modes:
            logging.warning(
                "Unknown intrabar_mode '%s' for %s %s. Falling back to hybrid_ohlc.",
                self.intrabar_mode,
                self.forex_pair,
                self.timeframe,
            )
            self.intrabar_mode = "hybrid_ohlc"

        if self.intrabar_mode != "lower_timeframe":
            return

        if intrabar_data is None or intrabar_data.empty:
            logging.warning(
                "Lower-timeframe mode requested but no intrabar_data provided for %s %s. Falling back to hybrid_ohlc.",
                self.forex_pair,
                self.timeframe,
            )
            self.intrabar_mode = "hybrid_ohlc"
            return

        required_cols = {"Timestamp", "High", "Low", "Close"}
        missing = required_cols.difference(intrabar_data.columns)
        if missing:
            logging.warning(
                "Intrabar data missing required columns %s for %s %s. Falling back to hybrid_ohlc.",
                sorted(missing),
                self.forex_pair,
                self.timeframe,
            )
            self.intrabar_mode = "hybrid_ohlc"
            return

        intrabar_df = intrabar_data.loc[:, ["Timestamp", "High", "Low", "Close"]].copy()
        intrabar_df["Timestamp"] = pd.to_datetime(
            intrabar_df["Timestamp"], errors="coerce"
        )
        intrabar_df["High"] = pd.to_numeric(intrabar_df["High"], errors="coerce")
        intrabar_df["Low"] = pd.to_numeric(intrabar_df["Low"], errors="coerce")
        intrabar_df["Close"] = pd.to_numeric(intrabar_df["Close"], errors="coerce")
        intrabar_df = intrabar_df.dropna(subset=["Timestamp", "High", "Low", "Close"])
        intrabar_df = intrabar_df.sort_values("Timestamp").reset_index(drop=True)

        if intrabar_df.empty:
            logging.warning(
                "Intrabar data became empty after cleaning for %s %s. Falling back to hybrid_ohlc.",
                self.forex_pair,
                self.timeframe,
            )
            self.intrabar_mode = "hybrid_ohlc"
            return

        self._intrabar_timestamps = intrabar_df["Timestamp"].to_numpy(
            dtype="datetime64[ns]"
        )
        self._intrabar_highs = intrabar_df["High"].to_numpy(dtype=float)
        self._intrabar_lows = intrabar_df["Low"].to_numpy(dtype=float)
        self._intrabar_closes = intrabar_df["Close"].to_numpy(dtype=float)

    def _bar_bounds(
        self, index: int, timestamps_dt: np.ndarray
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_ts = pd.Timestamp(timestamps_dt[index])
        if index + 1 < len(timestamps_dt):
            end_ts = pd.Timestamp(timestamps_dt[index + 1])
        else:
            end_ts = start_ts + self._parent_bar_delta

        if end_ts <= start_ts:
            end_ts = start_ts + pd.Timedelta(minutes=1)
        return start_ts, end_ts

    def _close_trade_if_exit(
        self,
        trade: Trade,
        exit_condition: str | None,
        exit_price: float | None,
        exit_timestamp,
        quote_to_usd_rate: float | None,
    ) -> bool:
        if not exit_condition:
            return False

        exit_quote_to_usd_rate = self._resolve_quote_to_usd_rate_for_exit(
            exit_timestamp, parent_fallback_rate=quote_to_usd_rate
        )
        trade.position_events.append((exit_timestamp, exit_condition))
        trade.close_trade(
            exit_timestamp,
            exit_price,
            self.commission_per_lot,
            exit_quote_to_usd_rate,
        )
        self._apply_closed_trade_balance(trade)
        return True

    def _evaluate_open_trade_exit_hybrid(
        self,
        trade: Trade,
        current_high: float,
        current_low: float,
        current_price: float,
        timestamp,
    ) -> tuple[str | None, float | None, pd.Timestamp]:
        position_manager = trade.plan.position_manager
        exit_result = position_manager.check_exit_intrabar(current_high, current_low)
        if exit_result:
            exit_condition, exit_price = exit_result
            return exit_condition, exit_price, pd.to_datetime(timestamp)

        exit_condition, exit_price, event = position_manager.update(current_price)
        if event:
            trade.position_events.append((timestamp, event))
        return exit_condition, exit_price, pd.to_datetime(timestamp)

    def _evaluate_open_trade_exit_lower_timeframe(
        self,
        trade: Trade,
        parent_start_ts: pd.Timestamp,
        parent_end_ts: pd.Timestamp,
        current_high: float,
        current_low: float,
        current_price: float,
        timestamp,
    ) -> tuple[str | None, float | None, pd.Timestamp]:
        if self._intrabar_timestamps is None:
            return self._evaluate_open_trade_exit_hybrid(
                trade,
                current_high=current_high,
                current_low=current_low,
                current_price=current_price,
                timestamp=timestamp,
            )

        start_np = np.datetime64(parent_start_ts)
        end_np = np.datetime64(parent_end_ts)
        left = int(np.searchsorted(self._intrabar_timestamps, start_np, side="left"))
        right = int(np.searchsorted(self._intrabar_timestamps, end_np, side="left"))

        if right <= left:
            return self._evaluate_open_trade_exit_hybrid(
                trade,
                current_high=current_high,
                current_low=current_low,
                current_price=current_price,
                timestamp=timestamp,
            )

        position_manager = trade.plan.position_manager
        for i in range(left, right):
            sub_high = self._intrabar_highs[i]
            sub_low = self._intrabar_lows[i]
            sub_close = self._intrabar_closes[i]
            sub_ts = pd.Timestamp(self._intrabar_timestamps[i])

            if not (
                np.isfinite(sub_high) and np.isfinite(sub_low) and np.isfinite(sub_close)
            ):
                continue

            exit_result = position_manager.check_exit_intrabar(sub_high, sub_low)
            if exit_result:
                exit_condition, exit_price = exit_result
                return exit_condition, exit_price, sub_ts

            # Keep trailing/breakeven updates aligned with actual sub-bar progression.
            exit_condition, exit_price, event = position_manager.update(sub_close)
            if event:
                trade.position_events.append((sub_ts, event))
            if exit_condition:
                return exit_condition, exit_price, sub_ts

        return None, None, pd.to_datetime(timestamp)

    def _apply_closed_trade_balance(self, trade: Trade) -> None:
        """
        Apply a closed trade's net result to account balance cumulatively.

        Important for multi-leg positions:
        - Each Trade object stores its own balance snapshot at entry.
        - On close, we must update the account balance by net delta, not overwrite
          with the trade object's local balance projection.
        """
        pnl = float(trade.pnl or 0.0)
        commission = float(trade.commission or 0.0)
        net_delta = pnl - commission
        self.balance = round(self.balance + net_delta, 2)
        trade.balance_after_trade = self.balance

    def _compute_unrealized_pnl(
        self, trade: Trade, mark_price: float, quote_to_usd_rate: float | None
    ) -> float:
        if trade.exit_price is not None:
            return 0.0
        if mark_price is None or not np.isfinite(mark_price):
            return 0.0
        if self.quote_currency != "USD" and (
            quote_to_usd_rate is None or not np.isfinite(quote_to_usd_rate)
        ):
            return 0.0

        net_pips = strategy_core.PositionCalculator.calculate_pip_change(
            self.forex_pair,
            trade.entry_price,
            float(mark_price),
            trade.direction,
        )
        pnl = strategy_core.PositionCalculator.calculate_profit_from_pips(
            self.forex_pair,
            net_pips,
            trade.units,
            float(mark_price),
            quote_to_usd_rate,
        )
        return float(pnl)

    def _compute_mark_to_market_equity(
        self, mark_price: float, quote_to_usd_rate: float | None
    ) -> float:
        unrealized = 0.0
        commission_if_closed_now = 0.0
        for trade in self.open_trades:
            unrealized += self._compute_unrealized_pnl(
                trade, mark_price, quote_to_usd_rate
            )
            commission_if_closed_now += trade.lot_size * self.commission_per_lot
        equity = self.balance + unrealized - commission_if_closed_now
        return round(float(equity), 2)

    def _record_equity_sample(
        self, timestamp, mark_price: float, quote_to_usd_rate: float | None
    ) -> None:
        equity = self._compute_mark_to_market_equity(mark_price, quote_to_usd_rate)
        ts = pd.to_datetime(timestamp)
        self.equity_samples.append((ts, equity))

        if equity > self.peak_balance:
            self.peak_balance = equity

        drawdown = self.peak_balance - equity
        drawdown_pct = (drawdown / self.peak_balance) * 100 if self.peak_balance > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_pct = drawdown_pct

    # def __repr__(self):
    #     return f"Backtester({self.strategy.NAME}, {self.forex_pair}, {self.timeframe}) | {self.strategy.PARAMETER_SETTINGS}"

    def _resolve_metrics_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Metrics/score can be computed on an evaluation sub-window while still running
        the simulation on a larger dataset (e.g., ML train+test load, test-only score).
        """
        raw_start = (
            pd.to_datetime(self.metrics_start_date)
            if self.metrics_start_date
            else pd.to_datetime(self.data_start_date)
        )
        raw_end = (
            pd.to_datetime(self.metrics_end_date)
            if self.metrics_end_date
            else pd.to_datetime(self.data_end_date)
        )
        start = raw_start.normalize()
        end = raw_end.normalize()
        if end < start:
            raise ValueError(
                f"Invalid metrics window: end {end.date()} < start {start.date()}."
            )
        return start, end

    @staticmethod
    def _slice_trades_by_entry_date(
        trades: list[Trade], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[Trade]:
        end_exclusive = end_date + pd.Timedelta(days=1)
        return [
            trade
            for trade in trades
            if start_date <= pd.to_datetime(trade.entry_timestamp) < end_exclusive
        ]

    def _compute_drawdown_from_trades(self, trades: list[Trade]) -> tuple[float, float]:
        """
        Recompute drawdown using only the provided trade subset.
        """
        if not trades:
            return 0.0, 0.0

        peak = self.initial_balance
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        for trade in trades:
            balance = trade.balance_after_trade
            if balance > peak:
                peak = balance
            drawdown = peak - balance
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        return max_drawdown, max_drawdown_pct

    def _compute_drawdown_from_equity_samples(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> tuple[float, float]:
        if not self.equity_samples:
            return 0.0, 0.0

        end_exclusive = end_date + pd.Timedelta(days=1)
        samples = [
            equity
            for ts, equity in self.equity_samples
            if start_date <= pd.to_datetime(ts) < end_exclusive
        ]
        if not samples:
            return 0.0, 0.0

        peak = self.initial_balance
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        for equity in samples:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        return max_drawdown, max_drawdown_pct

    def _calculate_metrics(self):
        metrics_start, metrics_end = self._resolve_metrics_bounds()
        trading_start_raw = pd.to_datetime(self.data["Timestamp"].iloc[self.trading_start_index])
        trading_start_effective = max(trading_start_raw, metrics_start)
        metric_trades = self._slice_trades_by_entry_date(
            self.all_trades, metrics_start, metrics_end
        )
        total_trades = len(metric_trades)

        # Set up the initial metrics dictionary, including new metrics.
        metrics = {
            "Data_Start_Date": metrics_start.strftime("%Y-%m-%d"),
            "Data_End_Date": metrics_end.strftime("%Y-%m-%d"),
            "Trading_Start_Date": trading_start_effective.strftime("%Y-%m-%d"),
            "Total_Trades": total_trades,
            "Winning_Trades": 0,
            "Gross_Profit": 0.0,
            "Gross_Loss": 0.0,
            "Net_Profit": 0.0,
            "Total_Return_Pct": 0.0,
            "Win_Rate": 0.0,
            "Profit_Factor": 0.0,
            "Max_Drawdown": 0.0,
            "Max_Drawdown_Pct": 0.0,
            "Average_Trade_Duration_Minutes": 0.0,
            "Initial_Balance": self.initial_balance,
            "Final_Balance": self.initial_balance,
            "Sharpe_Ratio": 0.0,
            "Sortino_Ratio": 0.0,
            "Calmar_Ratio": 0.0,
            "Recovery_Factor": 0.0,
            "Win_Loss_Ratio": 0.0,
            "Trade_Expectancy_Pct": 0.0,
            "Expectancy_Per_Day_Pct": 0.0,
            "Trade_Return_Std": 0.0,
            "Trades_Per_Day": 0.0,
            "Max_Consecutive_Wins": 0,
            "Max_Consecutive_Losses": 0,
            "Max_Margin_Required_Pct": 0.0,
        }

        # If no trades were executed, return default metrics.
        if total_trades == 0:
            metrics_df = pd.DataFrame([metrics])
            self.metrics_df = metrics_df
            return

        # Basic aggregated metrics.
        winning_trades = sum(
            1 for trade in metric_trades if trade.pnl is not None and trade.pnl > 0
        )
        gross_profit = sum(
            trade.pnl for trade in metric_trades if trade.pnl is not None and trade.pnl > 0
        )
        gross_loss = sum(
            trade.pnl for trade in metric_trades if trade.pnl is not None and trade.pnl < 0
        )
        initial_balance = self.initial_balance
        final_balance = metric_trades[-1].balance_after_trade
        net_profit = final_balance - initial_balance
        total_return_pct = (
            (net_profit / initial_balance) * 100 if initial_balance != 0 else 0
        )
        win_rate = winning_trades / total_trades
        profit_factor = (
            (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
        )
        average_trade_duration = (
            sum(trade.duration for trade in metric_trades if trade.duration is not None)
            / total_trades
        )

        # Compute trade returns (as percentages) for Sharpe, Sortino, and Trade Expectancy.
        trade_returns = [
            trade.return_pct for trade in metric_trades if trade.return_pct is not None
        ]
        mean_return = np.mean(trade_returns) if trade_returns else 0.0
        std_return = np.std(trade_returns) if trade_returns else 0.0
        sharpe_ratio = (mean_return / std_return) if std_return != 0 else float("nan")

        # Sortino Ratio: only use downside returns.
        downside_returns = [r for r in trade_returns if r < 0]
        std_downside = np.std(downside_returns) if downside_returns else 0.0
        sortino_ratio = (
            (mean_return / std_downside) if std_downside != 0 else float("inf")
        )

        # Drawdown and drawdown-based metrics in evaluation window only.
        # Prefer mark-to-market equity samples; fallback to closed-trade path if unavailable.
        max_drawdown, max_drawdown_pct = self._compute_drawdown_from_equity_samples(
            metrics_start, metrics_end
        )
        if max_drawdown == 0.0 and max_drawdown_pct == 0.0 and metric_trades:
            max_drawdown, max_drawdown_pct = self._compute_drawdown_from_trades(
                metric_trades
            )
        calmar_ratio = (
            (total_return_pct / max_drawdown_pct)
            if max_drawdown_pct != 0
            else float("inf")
        )
        recovery_factor = (
            (net_profit / max_drawdown) if max_drawdown != 0 else float("inf")
        )

        # Win/Loss Ratio (Average): based on percentage returns.
        wins_pct = [
            trade.return_pct
            for trade in metric_trades
            if trade.return_pct is not None and trade.return_pct > 0
        ]
        losses_pct = [
            trade.return_pct
            for trade in metric_trades
            if trade.return_pct is not None and trade.return_pct < 0
        ]
        avg_win_pct = np.mean(wins_pct) if wins_pct else 0.0
        avg_loss_pct = np.mean(losses_pct) if losses_pct else 0.0
        win_loss_ratio = (
            (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct != 0 else float("inf")
        )

        # Standard Deviation of Trade Returns.
        trade_return_std = std_return

        # Trades Per Day: compute using the metrics/evaluation window.
        num_days = (metrics_end - metrics_start).days + 1
        trades_per_day = total_trades / num_days if num_days > 0 else total_trades

        # Trade Expectancy as a percentage: expected return per trade.
        trade_expectancy_pct = (win_rate * avg_win_pct) - (
            (1 - win_rate) * abs(avg_loss_pct)
        )
        expectancy_per_day_pct = trade_expectancy_pct * trades_per_day

        # Maximum Consecutive Wins and Losses.
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        for trade in metric_trades:
            if trade.pnl is not None:
                if trade.pnl > 0:
                    current_wins += 1
                    current_losses = 0
                elif trade.pnl < 0:
                    current_losses += 1
                    current_wins = 0
                else:
                    current_wins = 0
                    current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
            max_consec_losses = max(max_consec_losses, current_losses)

        # Maximum Percentage Margin Required.
        max_pct_margin_required = max(
            (
                trade.margin_required / trade.balance_before_trade * 100
                for trade in metric_trades
                if trade.balance_before_trade > 0
            ),
            default=0.0,
        )

        # Update the metrics dictionary.
        metrics["Winning_Trades"] = winning_trades
        metrics["Gross_Profit"] = round(gross_profit, 2)
        metrics["Gross_Loss"] = round(gross_loss, 2)
        metrics["Net_Profit"] = round(net_profit, 2)
        metrics["Total_Return_Pct"] = round(total_return_pct, 2)
        metrics["Win_Rate"] = round(win_rate, 2)
        metrics["Profit_Factor"] = (
            round(profit_factor, 2) if profit_factor != float("inf") else profit_factor
        )
        metrics["Max_Drawdown"] = round(max_drawdown, 2)
        metrics["Max_Drawdown_Pct"] = round(max_drawdown_pct, 2)
        metrics["Average_Trade_Duration_Minutes"] = round(average_trade_duration, 2)
        metrics["Initial_Balance"] = round(initial_balance, 2)
        metrics["Final_Balance"] = round(final_balance, 2)
        metrics["Sharpe_Ratio"] = (
            round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else sharpe_ratio
        )
        metrics["Sortino_Ratio"] = (
            round(sortino_ratio, 2) if sortino_ratio != float("inf") else sortino_ratio
        )
        metrics["Calmar_Ratio"] = (
            round(calmar_ratio, 2) if calmar_ratio != float("inf") else calmar_ratio
        )
        metrics["Recovery_Factor"] = (
            round(recovery_factor, 2)
            if recovery_factor != float("inf")
            else recovery_factor
        )
        metrics["Win_Loss_Ratio"] = (
            round(win_loss_ratio, 2)
            if win_loss_ratio != float("inf")
            else win_loss_ratio
        )
        metrics["Trade_Expectancy_Pct"] = round(trade_expectancy_pct, 2)
        metrics["Expectancy_Per_Day_Pct"] = round(expectancy_per_day_pct, 2)
        metrics["Trade_Return_Std"] = round(trade_return_std, 4)
        metrics["Trades_Per_Day"] = round(trades_per_day, 2)
        metrics["Max_Consecutive_Wins"] = max_consec_wins
        metrics["Max_Consecutive_Losses"] = max_consec_losses
        metrics["Max_Margin_Required_Pct"] = round(max_pct_margin_required, 2)

        metrics_df = pd.DataFrame([metrics])
        self.metrics_df = metrics_df
        return

    def initialize_sqlhelper(self):
        self.backtest_sqlhelper = BacktestSQLHelper()

    def get_data(self):
        return self.data

    def run_backtest(self):
        """
        Executes the backtest using the provided strategy.
        Now uses generate_trade_plan() from Strategy to determine all trade logic.
        """
        closes = self.data["Close"].values
        highs = self.data["High"].values
        lows = self.data["Low"].values
        timestamps = self.data["Timestamp"].values
        timestamps_dt = pd.to_datetime(self.data["Timestamp"]).to_numpy(
            dtype="datetime64[ns]"
        )
        quote_to_usd_rates = (
            self.quote_to_usd_data["Close"].values
            if self.quote_to_usd_data is not None
            else None
        )

        for index in range(len(self.data)):

            # Skip the first 100 rows so we have a semblance of historical data
            if index < self.trading_start_index:
                continue

            current_price = closes[index]
            current_high = highs[index]
            current_low = lows[index]
            timestamp = timestamps[index]
            entry_allowed_on_bar = (
                True
                if self._entry_allowed_mask is None
                else bool(self._entry_allowed_mask[index])
            )

            quote_to_usd_rate = None
            if quote_to_usd_rates is not None:
                if index < len(quote_to_usd_rates):
                    quote_to_usd_rate = quote_to_usd_rates[index]
                if quote_to_usd_rate is None or not np.isfinite(quote_to_usd_rate):
                    quote_to_usd_rate = self._lookup_rate_asof(
                        self._quote_to_usd_parent_timestamps,
                        self._quote_to_usd_parent_values,
                        timestamp,
                        fallback=None,
                    )

            # Generate zero or more trade plans (entries and/or exits) using strategy logic
            # Out of entry-session windows, skip strategy calls only when flat to avoid
            # mutating strategy entry-state on bars where entries cannot execute.
            if entry_allowed_on_bar or self.open_trades:
                plans: list[strategy_core.TradePlan] = (
                    self.strategy.generate_trade_plan(
                        # Slice precomputed dataset up to current index to mimic real-time data
                        index,
                        self.position,
                        self.balance,
                        quote_to_usd_rate,
                    )
                )
            else:
                plans = []

            # --- Step 1: Check SL/TP exit for open trades BEFORE executing any strategy ENTRY/EXIT ---
            exited_trades = []
            bar_start_ts, bar_end_ts = self._bar_bounds(index, timestamps_dt)
            for trade in self.open_trades:
                if self.intrabar_mode == "lower_timeframe":
                    exit_condition, exit_price, exit_timestamp = (
                        self._evaluate_open_trade_exit_lower_timeframe(
                            trade,
                            parent_start_ts=bar_start_ts,
                            parent_end_ts=bar_end_ts,
                            current_high=current_high,
                            current_low=current_low,
                            current_price=current_price,
                            timestamp=timestamp,
                        )
                    )
                else:
                    exit_condition, exit_price, exit_timestamp = (
                        self._evaluate_open_trade_exit_hybrid(
                            trade,
                            current_high=current_high,
                            current_low=current_low,
                            current_price=current_price,
                            timestamp=timestamp,
                        )
                    )

                if self._close_trade_if_exit(
                    trade,
                    exit_condition=exit_condition,
                    exit_price=exit_price,
                    exit_timestamp=exit_timestamp,
                    quote_to_usd_rate=quote_to_usd_rate,
                ):
                    exited_trades.append(trade)

            # Remove only trades that exited
            self.open_trades = [t for t in self.open_trades if t not in exited_trades]

            # --- Step 2: Execute new trade plans (entries + strategy EXITs)
            # Keep bar-start position semantics so strategy-exit/SLTP handling remains
            # behaviorally compatible while still allowing multi-leg entries on flat bars.
            position_at_bar_start = self.position
            exited_by_strategy_plan = False

            for plan in plans:
                # Exit logic (from EXIT StrategyPlan) — if we're holding a position
                if plan.tag == "EXIT":
                    if self.open_trades:
                        exit_price = (
                            current_price * (1 - self.slippage)
                            if self.position == 1
                            else current_price * (1 + self.slippage)
                        )
                        exit_quote_to_usd_rate = self._resolve_quote_to_usd_rate_for_exit(
                            timestamp, parent_fallback_rate=quote_to_usd_rate
                        )
                        for trade in self.open_trades:
                            trade.close_trade(
                                timestamp,
                                exit_price,
                                self.commission_per_lot,
                                exit_quote_to_usd_rate,
                            )
                            self._apply_closed_trade_balance(trade)
                        self.open_trades = []
                        exited_by_strategy_plan = True
                    continue

                # Preserve original no-same-bar-reentry behavior:
                # entries are only allowed if the bar started flat and no strategy EXIT
                # has already been consumed on this bar.
                if position_at_bar_start != 0 or exited_by_strategy_plan:
                    continue
                if not entry_allowed_on_bar:
                    continue

                # Skip if any critical values are missing or invalid
                if any(
                    math.isnan(x) for x in [plan.entry_price, plan.stop_loss]
                ) or plan.direction not in ["BUY", "SELL"]:
                    continue

                # Allow stacking same-direction legs (e.g., NNFX TP1 + Runner),
                # but never open opposing directions in the same bar.
                existing_direction = (
                    self.open_trades[0].direction if self.open_trades else None
                )
                if existing_direction is not None and plan.direction != existing_direction:
                    continue

                entry_price = (
                    plan.entry_price * (1 + self.slippage)
                    if plan.direction == "BUY"
                    else plan.entry_price * (1 - self.slippage)
                )

                # Ensure we have enough margin to open this trade
                margin_required = strategy_core.PositionCalculator.calculate_required_margin(
                    self.forex_pair,
                    plan.units,
                    self.leverage,
                    entry_price,
                    quote_to_usd_rate,
                )
                if margin_required > self.margin_remaining:
                    continue  # Skip this plan if insufficient margin

                # Estimate pip value
                pip_value = strategy_core.PositionCalculator.calculate_current_pip_value(
                    self.forex_pair, plan.units, entry_price, quote_to_usd_rate
                )

                # Create the Trade object and attach the original StrategyPlan for context
                trade = Trade(
                    plan=plan,
                    forex_pair=self.forex_pair,
                    entry_timestamp=timestamp,
                    entry_price=entry_price,
                    balance=self.balance,
                    leverage=self.leverage,
                    margin_required=margin_required,
                    pip_value=pip_value,
                )

                self.all_trades.append(trade)
                self.open_trades.append(trade)

            # Mark-to-market equity and realistic drawdown with open-trade PnL included.
            self._record_equity_sample(timestamp, current_price, quote_to_usd_rate)

            # Automatically set position based on current open trades
            if not self.open_trades:
                self.position = 0
            elif self.open_trades[0].direction == "BUY":
                self.position = 1
            elif self.open_trades[0].direction == "SELL":
                self.position = -1
            assert self.position in [
                -1,
                0,
                1,
            ], f"Unexpected position value: {self.position}"

        # Calculate metrics from trades
        self._calculate_metrics()
        return

    def get_number_of_trades(self):
        return len(self.all_trades)

    def get_metrics_df(self):
        return self.metrics_df

    def set_metrics_df(self, metrics_df):
        self.metrics_df = metrics_df

    def get_metric(self, metric_name: str):
        # Since the metrics_df is a one-row DataFrame, just grab the first/only value of the column
        value = self.metrics_df[metric_name].iloc[0]
        return value

    def save_trades(self):
        """
        Save the backtest results to a SQLite database.
        """
        # Build a dictionary of trade data including the extra dimension fields
        trade_dict = {
            "Backtest_ID": [self.run_id for _ in self.all_trades],
            "Timestamp": [trade.entry_timestamp for trade in self.all_trades],
            "Exit_Timestamp": [trade.exit_timestamp for trade in self.all_trades],
            "Direction": [
                1 if trade.direction == "BUY" else -1 for trade in self.all_trades
            ],
            "Entry_Price": [trade.entry_price for trade in self.all_trades],
            "Units": [trade.units for trade in self.all_trades],
            "Exit_Price": [trade.exit_price for trade in self.all_trades],
            "PnL": [trade.pnl for trade in self.all_trades],
            "Net_Pips": [trade.net_pips for trade in self.all_trades],
            "Commission": [trade.commission for trade in self.all_trades],
            "Return_Pct": [trade.return_pct for trade in self.all_trades],
            "Duration_Minutes": [trade.duration for trade in self.all_trades],
            "Starting_Balance": [
                trade.balance_before_trade for trade in self.all_trades
            ],
            "End_Balance": [trade.balance_after_trade for trade in self.all_trades],
            "Leverage": [trade.leverage for trade in self.all_trades],
            "Margin_Required": [trade.margin_required for trade in self.all_trades],
            "Margin_Required_Pct": [
                round(trade.margin_required / trade.balance_before_trade, 2)
                for trade in self.all_trades
            ],
            "Pip_Value": [trade.pip_value for trade in self.all_trades],
            "Plan_Source": [trade.plan.source for trade in self.all_trades],
            "Position_Events": [
                str(trade.position_events) for trade in self.all_trades
            ],
        }

        dataframe = pd.DataFrame(trade_dict)

        self.backtest_sqlhelper.insert_trades(dataframe)

        return dataframe

    # Previous scoring function retained for reference.
    # def calculate_composite_score(self):
    #     expectancy_per_day = self.get_metric("Expectancy_Per_Day_Pct")
    #     trade_expectancy = self.get_metric("Trade_Expectancy_Pct")
    #     profit_factor = min(self.get_metric("Profit_Factor"), 5.0)
    #     win_loss_ratio = min(self.get_metric("Win_Loss_Ratio"), 3.0)
    #     trades_per_day = self.get_metric("Trades_Per_Day")
    #
    #     if trades_per_day < 0.02:
    #         trade_expectancy *= 0.5
    #
    #     score = (
    #         expectancy_per_day * 2.5
    #         + trade_expectancy * 0.75
    #         + profit_factor * 0.5
    #         + win_loss_ratio * 0.5
    #         + trades_per_day * 0.25
    #     )
    #
    #     net_profit = self.get_metric("Net_Profit")
    #     score += math.log1p(max(net_profit, 0)) * 0.3
    #     score += max(net_profit, 0) / 250
    #
    #     max_drawdown_pct = self.get_metric("Max_Drawdown_Pct")
    #     trade_std = self.get_metric("Trade_Return_Std")
    #     timeframe = self.timeframe
    #     trades_per_day_threshold = config.MIN_TRADES_PER_DAY[timeframe]
    #     ratio = trades_per_day / trades_per_day_threshold
    #
    #     if trades_per_day >= trades_per_day_threshold:
    #         activity_penalty = 1.0
    #     elif ratio <= 0:
    #         activity_penalty = 0.0
    #     elif timeframe in ["1_day", "4_hour", "2_hour"]:
    #         activity_penalty = ratio**0.25
    #     else:
    #         activity_penalty = 1 / (1 + (1 / max(ratio, 1e-6)) ** 2)
    #
    #     score *= activity_penalty
    #     score *= 1 / (1 + (max_drawdown_pct / 40) ** 1.5)
    #     score *= 1 / (1 + trade_std / 20)
    #     return round(score, 4)

    def calculate_composite_score(self):
        """
        Calculates a composite score for evaluating strategy performance,
        incorporating profitability, risk, consistency, and trade activity.
        Applies safeguards for low sample sizes and score inflation.
        """

        # --- Primary performance metrics ---
        expectancy_per_day = self.get_metric("Expectancy_Per_Day_Pct")
        trade_expectancy = self.get_metric("Trade_Expectancy_Pct")
        profit_factor = min(self.get_metric("Profit_Factor"), 5.0)  # Cap extreme values
        win_loss_ratio = min(
            self.get_metric("Win_Loss_Ratio"), 3.0
        )  # Cap skewed outcomes
        trades_per_day = self.get_metric("Trades_Per_Day")

        # Reduce the influence of trade expectancy if trading frequency is very low
        if trades_per_day < 0.02:
            trade_expectancy *= 0.5  # Avoid inflating score from 1–2 good trades

        # --- Base score calculation ---
        # Weighted sum of core performance metrics
        score = (
            expectancy_per_day * 2.5  # Daily compounding return
            + trade_expectancy * 0.75  # Average per-trade quality
            + profit_factor * 0.5  # Risk-reward efficiency
            + win_loss_ratio * 0.5  # Reward/risk asymmetry
            + trades_per_day * 0.25  # Mild boost for active systems
        )

        # --- Profitability metrics ---
        net_profit = self.get_metric("Net_Profit")

        # Previous profit terms (kept for reference):
        # score += math.log1p(max(net_profit, 0)) * 0.3
        # score += max(net_profit, 0) / 250

        # Light rebalance:
        # - keep profit important,
        # - avoid runaway dominance from one large net-profit outlier.
        score += math.log1p(max(net_profit, 0)) * 0.3
        score += min(max(net_profit, 0) / 500, 60.0)

        # --- Risk and volatility metrics ---
        max_drawdown_pct = self.get_metric("Max_Drawdown_Pct")
        trade_std = self.get_metric("Trade_Return_Std")
        sharpe_ratio = self.get_metric("Sharpe_Ratio")
        sortino_ratio = self.get_metric("Sortino_Ratio")
        max_margin_required_pct = self.get_metric("Max_Margin_Required_Pct")
        timeframe = self.timeframe

        # --- Trade count context ---
        total_trades = self.get_metric("Total_Trades")
        start = pd.to_datetime(self.get_metric("Data_Start_Date"))
        end = pd.to_datetime(self.get_metric("Data_End_Date"))
        backtest_days = (end - start).days

        # Require at least ~5 trades per year to allow bonus metrics
        min_bonus_trades = max(20, backtest_days / 365 * 5)

        # --- Activity penalty based on expected trading rate ---
        trades_per_day_threshold = config.MIN_TRADES_PER_DAY[timeframe]
        ratio = trades_per_day / trades_per_day_threshold

        if trades_per_day >= trades_per_day_threshold:
            activity_penalty = 1.0  # No penalty
        elif ratio <= 0:
            activity_penalty = 0.0  # No activity = no score
        elif timeframe in ["1_day", "4_hour", "2_hour"]:
            # Moderate decay for high-timeframe strategies
            activity_penalty = ratio**0.25
        else:
            # Sharper decay for intraday strategies
            activity_penalty = 1 / (1 + (1 / max(ratio, 1e-6)) ** 2)

        score *= activity_penalty  # Apply activity-based penalty

        # --- Penalize excessive drawdown (exponential) ---
        # 20% drawdown → 50% reduction, 40% → 80%, etc.
        score *= 1 / (1 + (max_drawdown_pct / 40) ** 1.5)

        # --- Penalize high return volatility ---
        # 0% std = no penalty; 20% std = 0.33; 40% std = 0.2
        score *= 1 / (1 + trade_std / 20)

        # --- Bonus for Sharpe/Sortino only if trade count is adequate ---
        # if total_trades >= min_bonus_trades:
        #     # Scale bonus by how confident we are (up to 1.0)
        #     confidence_weight = min(1.0, total_trades / (min_bonus_trades * 2))

        #     if sharpe_ratio and sharpe_ratio != float("inf") and sharpe_ratio > 1.0:
        #         score += min((sharpe_ratio - 1.0), 3.0) * 0.25 * confidence_weight

        #     if sortino_ratio and sortino_ratio != float("inf") and sortino_ratio > 2.0:
        #         score += min((sortino_ratio - 2.0), 3.0) * 0.25 * confidence_weight

        # --- Leverage / position sizing penalty ---
        # if max_margin_required_pct > 80:
        #     score *= 0.85  # Flat penalty for overly aggressive margin usage

        # --- Final normalization and clamping ---
        # score = round(min(score * 10, 20.0), 4)  # Clamp to max score of 20

        score = round(score, 4)

        return score


# Example Usage
if __name__ == "__main__":
    forex_pair = "USDJPY"  # Example forex pair
    strategy = strategy_core.EMACross({"ema1": 9, "ema2": 21})
    backtester = Backtester(
        strategy_core.EMACross,
        forex_pair,
        "4_hour",
        initial_balance=10_000,
        leverage=50,
    )

    # test_price = 149.8300
    # quote_to_usd_rate = 0.0067
    # units = 100000
    # x = PositionCalculator.calculate_trade_units(forex_pair, 50_236, 0.02, test_price, 0.00864, quote_to_usd_rate)
    # y = PositionCalculator.calculate_current_pip_value(forex_pair, x, test_price, quote_to_usd_rate)
    # z = PositionCalculator.calculate_required_margin(forex_pair, x, 50, test_price, quote_to_usd_rate)
    # print(x, y, z)

    backtester.run_backtest()

    backtester.save_run()
