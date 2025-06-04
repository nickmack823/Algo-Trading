import math

import numpy as np
import pandas as pd

from scripts import config, strategies
from scripts.sql import BacktestSQLHelper, HistoricalDataSQLHelper


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


class Trade:
    """
    A class to store details of a trade, including entry, exit, PnL, and commissions.
    """

    def __init__(
        self,
        forex_pair: str,
        entry_timestamp,
        trade_type: str,
        entry_price: float,
        units: int,
        balance: float,
        leverage: int,
        margin_required: float,
        pip_value: float,
    ):
        self.forex_pair = forex_pair
        self.entry_timestamp = entry_timestamp  # Time of trade entry
        self.trade_type = trade_type  # 'BUY' or 'SELL'
        self.entry_price = entry_price  # Price at entry
        self.units = units  # Size of the trade in units (100,000 units/lot)
        self.balance_before_trade = balance  # Account balance before the trade
        self.balance_after_trade = balance  # Account balance after the trade
        self.leverage = leverage  # Leverage factor
        self.margin_required = margin_required  # Margin required for the trade
        self.pip_value = pip_value  # Pip value of the trade

        self.num_decimals = 3 if "JPY" in self.forex_pair else 5
        self.entry_price = round(self.entry_price, self.num_decimals)

        self.lot_size = round(units / 100_000, 2)  # Size of the trade in lots

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
    ):
        if self.exit_price is not None:
            return  # Prevent closing a trade twice

        self.exit_price = exit_price

        self.net_pips = PositionCalculator.calculate_pip_change(
            self.forex_pair, self.entry_price, self.exit_price, self.trade_type
        )

        # Calculate PnL based on trade direction
        self.pnl = PositionCalculator.calculate_profit_from_pips(
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
            f"{self.entry_timestamp}: {self.trade_type} @ {self.entry_price}, "
            f"Lot Size: {self.lot_size}, Exit: {self.exit_price}, "
            f"PnL: ${self.pnl}, Pip Change: {self.net_pips}, Commission: {self.commission}, "
            f"Return %: {self.return_pct}, Duration (min): {self.duration}, "
            f"Balance: ${self.balance_after_trade}, Leverage: {self.leverage}, Margin Required: ${self.margin_required}, Pip Value: {self.pip_value}"
        )


class Backtester:
    def __init__(
        self,
        strategy: strategies.BaseStrategy,
        forex_pair: str,
        timeframe: str,
        initial_balance=10_000,
        risk_per_trade=0.02,
        commission_per_lot=6,
        slippage=0.0002,
        leverage=50,
    ):
        """
        Initializes the backtester with historical data and trading parameters.
        """
        self.strategy = strategy  # Trading strategy
        self.forex_pair = forex_pair  # Forex pair being traded
        self.base_currency, self.quote_currency = (
            self.forex_pair[:3],
            self.forex_pair[3:],
        )

        self.timeframe = timeframe

        self.initial_balance = initial_balance  # Starting account balance
        self.balance = initial_balance  # Current account balance

        self.risk_per_trade = risk_per_trade  # % account balance risked per trade

        self.atr_multiplier = 1.5

        self.commission_per_lot = commission_per_lot  # Commission charged per lot (USD)
        self.slippage = slippage  # Slippage factor
        self.leverage = leverage  # Leverage factor

        # Keep track of margin remaining
        self.margin_remaining = initial_balance

        # Create a new instance of the BacktestSQLHelper
        self.backtest_sqlhelper = BacktestSQLHelper(
            f"{config.BACKTESTING_FOLDER}/{config.BACKTESTING_DB_NAME}"
        )
        self.forex_pair_id = self.backtest_sqlhelper.select_forex_pair_id(
            f"{self.base_currency}/{self.quote_currency}"
        )
        self.strategy_config_id = self.backtest_sqlhelper.insert_strategy_configuration(
            self.strategy.NAME,
            self.strategy.DESCRIPTION,
            self.strategy.PARAMETER_SETTINGS,
        )

        # Get historical data
        data_sqlhelper = HistoricalDataSQLHelper(
            f"{config.DATA_FOLDER}/{forex_pair}.db"
        )
        self.data = data_sqlhelper.get_historical_data(
            table=timeframe, columns=self.strategy.REQUIRED_COLUMNS
        )
        self.data_start_date, self.data_end_date = (
            self.data["Timestamp"].min(),
            self.data["Timestamp"].max(),
        )

        # If USD not quote, get secondary data for pip value & margin calculations
        self.quote_to_usd_data = None
        if self.quote_currency != "USD":
            secondary_data_sqlhelper = HistoricalDataSQLHelper(
                f"{config.DATA_FOLDER}/{self.quote_currency}USD.db"
            )
            self.quote_to_usd_data = secondary_data_sqlhelper.get_historical_data(
                table=timeframe
            )

        self.trades = []  # List to store executed trades
        self.position = 0  # Current position: +1 for long, -1 for short, 0 for none
        self.entry_trade = None  # Currently open trade
        self.max_drawdown = 0  # Maximum observed drawdown
        self.peak_balance = initial_balance  # Highest account balance observed

    def run_backtest(self):
        """
        Executes the backtest using the provided strategy.
        """
        bars_done = 0
        for index, row in self.data.iterrows():

            timestamp = row["Timestamp"]
            atr = row["ATR"]

            # Skip rows where ATR is nan
            if math.isnan(atr):
                continue

            # If USD not in pair, get secondary data for pip value calculations
            quote_to_usd_rate = None
            if self.quote_to_usd_data is not None:
                quote_to_usd_rate = self.quote_to_usd_data.loc[index]["Close"]

            # Generate trading signal
            signals = self.strategy.generate_signals(row, self.position)
            entry_long = self.strategy.ENTER_LONG in signals
            exit_long = self.strategy.EXIT_LONG in signals
            entry_short = self.strategy.ENTER_SHORT in signals
            exit_short = self.strategy.EXIT_SHORT in signals

            # Exit Long Position
            if self.position == 1 and exit_long and self.entry_trade:
                exit_price = row["Close"] * (1 - self.slippage)
                self.entry_trade.close_trade(
                    timestamp, exit_price, self.commission_per_lot, quote_to_usd_rate
                )
                self.balance = self.entry_trade.balance_after_trade
                self.position = 0
                self.entry_trade = None
            # Exit Short Position
            elif self.position == -1 and exit_short and self.entry_trade:
                exit_price = row["Close"] * (1 + self.slippage)
                self.entry_trade.close_trade(
                    timestamp, exit_price, self.commission_per_lot, quote_to_usd_rate
                )
                self.balance = self.entry_trade.balance_after_trade
                self.position = 0
                self.entry_trade = None

            # Enter Long Position
            if self.position == 0 and entry_long:

                entry_price = row["Close"] * (1 + self.slippage)
                trade_units = PositionCalculator.calculate_trade_units(
                    self.forex_pair,
                    self.balance,
                    self.risk_per_trade,
                    entry_price,
                    atr,
                    quote_to_usd_rate,
                )
                margin_required = PositionCalculator.calculate_required_margin(
                    self.forex_pair,
                    trade_units,
                    self.leverage,
                    entry_price,
                    quote_to_usd_rate,
                )
                if margin_required > self.margin_remaining:
                    continue

                pip_value = PositionCalculator.calculate_current_pip_value(
                    self.forex_pair, trade_units, entry_price, quote_to_usd_rate
                )

                self.entry_trade = Trade(
                    self.forex_pair,
                    timestamp,
                    "BUY",
                    entry_price,
                    trade_units,
                    self.balance,
                    self.leverage,
                    margin_required,
                    pip_value,
                )
                self.trades.append(self.entry_trade)
                self.position = 1

            # Enter Short Position
            elif self.position == 0 and entry_short:

                entry_price = row["Close"] * (1 - self.slippage)
                trade_units = PositionCalculator.calculate_trade_units(
                    self.forex_pair,
                    self.balance,
                    self.risk_per_trade,
                    entry_price,
                    atr,
                    quote_to_usd_rate,
                )
                margin_required = PositionCalculator.calculate_required_margin(
                    self.forex_pair,
                    trade_units,
                    self.leverage,
                    entry_price,
                    quote_to_usd_rate,
                )
                if margin_required > self.margin_remaining:
                    continue

                pip_value = PositionCalculator.calculate_current_pip_value(
                    self.forex_pair, trade_units, entry_price, quote_to_usd_rate
                )

                self.entry_trade = Trade(
                    self.forex_pair,
                    timestamp,
                    "SELL",
                    entry_price,
                    trade_units,
                    self.balance,
                    self.leverage,
                    margin_required,
                    pip_value,
                )
                self.trades.append(self.entry_trade)
                self.position = -1

            # Track drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

            drawdown = self.peak_balance - self.balance
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def save_run(self) -> int:
        """
        Calculate aggregated metrics for the entire backtest run using the Trade objects in self.trades.

        Returns:
            int: The run_id of the newly inserted backtest run.
        """
        total_trades = len(self.trades)

        # Set up the initial metrics dictionary, including new metrics.
        metrics = {
            "Timeframe": self.timeframe,
            "Data_Start_Date": self.data_start_date,
            "Data_End_Date": self.data_end_date,
            "Total_Trades": total_trades,
            "Winning_Trades": 0,
            "Gross_Profit": 0.0,
            "Gross_Loss": 0.0,
            "Net_Profit": 0.0,
            "Total_Return_Pct": 0.0,
            "Win_Rate": 0.0,
            "Profit_Factor": 0.0,
            "Max_Drawdown": getattr(self, "max_drawdown", 0.0),
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
            "Max_Pct_Margin_Required": 0.0,
        }

        # If no trades were executed, insert default metrics.
        if total_trades == 0:
            metrics_df = pd.DataFrame([metrics])
            self.run_id = self.backtest_sqlhelper.insert_backtest_run(
                self.forex_pair_id, self.strategy_config_id, metrics_df
            )
            return self.run_id

        # Basic aggregated metrics.
        winning_trades = sum(
            1 for trade in self.trades if trade.pnl is not None and trade.pnl > 0
        )
        gross_profit = sum(
            trade.pnl
            for trade in self.trades
            if trade.pnl is not None and trade.pnl > 0
        )
        gross_loss = sum(
            trade.pnl
            for trade in self.trades
            if trade.pnl is not None and trade.pnl < 0
        )
        initial_balance = self.initial_balance
        final_balance = self.trades[-1].balance_after_trade
        net_profit = final_balance - initial_balance
        total_return_pct = (
            (net_profit / initial_balance) * 100 if initial_balance != 0 else 0
        )
        win_rate = winning_trades / total_trades
        profit_factor = (
            (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
        )
        max_drawdown = getattr(self, "max_drawdown", 0.0)
        average_trade_duration = (
            sum(trade.duration for trade in self.trades if trade.duration is not None)
            / total_trades
        )

        # Compute trade returns (as percentages) for Sharpe, Sortino, and Trade Expectancy.
        trade_returns = [
            trade.return_pct for trade in self.trades if trade.return_pct is not None
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

        # Calmar Ratio & Recovery Factor: computed as net profit divided by max drawdown.
        calmar_ratio = (
            (net_profit / max_drawdown) if max_drawdown != 0 else float("inf")
        )
        recovery_factor = (
            (net_profit / max_drawdown) if max_drawdown != 0 else float("inf")
        )

        # Win/Loss Ratio (Average): based on percentage returns.
        wins_pct = [
            trade.return_pct
            for trade in self.trades
            if trade.return_pct is not None and trade.return_pct > 0
        ]
        losses_pct = [
            trade.return_pct
            for trade in self.trades
            if trade.return_pct is not None and trade.return_pct < 0
        ]
        avg_win_pct = np.mean(wins_pct) if wins_pct else 0.0
        avg_loss_pct = np.mean(losses_pct) if losses_pct else 0.0
        win_loss_ratio = (
            (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct != 0 else float("inf")
        )

        # Standard Deviation of Trade Returns.
        trade_return_std = std_return

        # Trades Per Day: compute using the data start and end dates.
        start_date = pd.to_datetime(self.data_start_date)
        end_date = pd.to_datetime(self.data_end_date)
        num_days = (end_date - start_date).days + 1
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
        for trade in self.trades:
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
                for trade in self.trades
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
        metrics["Max_Pct_Margin_Required"] = round(max_pct_margin_required, 2)

        metrics_df = pd.DataFrame([metrics])
        self.run_id = self.backtest_sqlhelper.insert_backtest_run(
            self.forex_pair_id, self.strategy_config_id, metrics_df
        )
        return self.run_id

    def save_trades(self):
        """
        Save the backtest results to a SQLite database.
        """
        # Build a dictionary of trade data including the extra dimension fields
        trade_dict = {
            "Backtest_ID": [self.run_id for _ in self.trades],
            "Timestamp": [trade.entry_timestamp for trade in self.trades],
            "Exit_Timestamp": [trade.exit_timestamp for trade in self.trades],
            "Direction": [
                1 if trade.trade_type == "BUY" else -1 for trade in self.trades
            ],
            "Entry_Price": [trade.entry_price for trade in self.trades],
            "Units": [trade.units for trade in self.trades],
            "Exit_Price": [trade.exit_price for trade in self.trades],
            "PnL": [trade.pnl for trade in self.trades],
            "Net_Pips": [trade.net_pips for trade in self.trades],
            "Commission": [trade.commission for trade in self.trades],
            "Return_Pct": [trade.return_pct for trade in self.trades],
            "Duration_Minutes": [trade.duration for trade in self.trades],
            "Starting_Balance": [trade.balance_before_trade for trade in self.trades],
            "End_Balance": [trade.balance_after_trade for trade in self.trades],
            "Leverage": [trade.leverage for trade in self.trades],
            "Margin_Required": [trade.margin_required for trade in self.trades],
            "Margin_Required_Pct": [
                round(trade.margin_required / trade.balance_before_trade, 2)
                for trade in self.trades
            ],
            "Pip_Value": [trade.pip_value for trade in self.trades],
        }

        dataframe = pd.DataFrame(trade_dict)

        self.backtest_sqlhelper.insert_trades(dataframe)


# Example Usage
if __name__ == "__main__":
    forex_pair = "USDJPY"  # Example forex pair
    strategy = strategies.EMACross({"ema1": 9, "ema2": 21})
    backtester = Backtester(
        strategies.EMACross, forex_pair, "4_hour", initial_balance=10_000, leverage=50
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
