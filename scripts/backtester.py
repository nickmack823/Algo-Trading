import math

import numpy as np
import pandas as pd

from scripts import config, strategies
from scripts.sql import BacktestSQLHelper, HistoricalDataSQLHelper


class Trade:
    """
    A class to store details of a trade, including entry, exit, PnL, and commissions.
    """

    def __init__(
        self,
        plan: strategies.TradePlan,
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

        self.net_pips = strategies.PositionCalculator.calculate_pip_change(
            self.forex_pair, self.entry_price, self.exit_price, self.direction
        )

        # Calculate PnL based on trade direction
        self.pnl = strategies.PositionCalculator.calculate_profit_from_pips(
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
        strategy: strategies.BaseStrategy,
        forex_pair: str,  # e.g. "EURUSD" with no slashes
        timeframe: str,
        initial_balance=10_000,
        commission_per_lot=6,
        slippage=0.0002,
        leverage=50,
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
        self.initial_balance = initial_balance  # Starting account balance
        self.balance = initial_balance  # Current account balance
        self.commission_per_lot = commission_per_lot  # Commission charged per lot (USD)
        self.slippage = slippage  # Slippage factor
        self.leverage = leverage  # Leverage factor

        # Keep track of margin remaining
        self.margin_remaining = initial_balance

        # SQL Helpers
        self.backtest_sqlhelper = BacktestSQLHelper()
        self.forex_pair_id = self.backtest_sqlhelper.get_forex_pair_id(
            f"{self.base_currency}/{self.quote_currency}"
        )
        self.strategy_config_id = self.backtest_sqlhelper.upsert_strategy_configuration(
            self.strategy.NAME,
            self.strategy.DESCRIPTION,
            self.strategy.PARAMETER_SETTINGS,
        )

        # Load historical OHLCV + indicator data
        data_sqlhelper = HistoricalDataSQLHelper(
            f"{config.DATA_FOLDER}/{self.forex_pair}.db"
        )
        self.data: pd.DataFrame = data_sqlhelper.get_historical_data(table=timeframe)
        self.data_start_date, self.data_end_date = (
            self.data["Timestamp"].min(),
            self.data["Timestamp"].max(),
        )
        # Precompute indicators
        self.strategy.prepare_data(self.data)

        # Set a start index for when we start trading on the data (to create a semblance of historical data)
        self.trading_start_index = 100

        # Load conversion data if non-USD quote currency
        self.quote_to_usd_data = None
        if self.quote_currency != "USD":
            secondary_data_sqlhelper = HistoricalDataSQLHelper(
                f"{config.DATA_FOLDER}/{self.quote_currency}USD.db"
            )
            self.quote_to_usd_data: pd.DataFrame = (
                secondary_data_sqlhelper.get_historical_data(table=timeframe)
            )

            # Set Timestamp as index for easier alignment
            quote_df = self.quote_to_usd_data.set_index("Timestamp").sort_index()
            data_timestamps = pd.Series(self.data["Timestamp"].unique())

            # Find missing timestamps
            missing_timestamps = data_timestamps[~data_timestamps.isin(quote_df.index)]

            # For each missing timestamp, forward-fill from the last known quote row
            if not missing_timestamps.empty:
                # Reindex the quote data to include missing timestamps
                all_timestamps = quote_df.index.union(missing_timestamps).sort_values()
                quote_df = quote_df.reindex(all_timestamps)

                # Forward-fill missing values
                quote_df = quote_df.ffill()

            # Reset index to have Timestamp as a column again
            self.quote_to_usd_data = quote_df.reset_index()

            secondary_data_sqlhelper.close_connection()
            secondary_data_sqlhelper = None

        self.all_trades: list[Trade] = []  # List of all executed trades
        self.open_trades: list[Trade] = []  # Stores open trades
        self.position = 0  # Current position: 1 for long, -1 for short, 0 for none
        self.max_drawdown = 0  # Maximum observed drawdown
        self.max_drawdown_pct = 0  # Maximum observed drawdown as a percentage
        self.peak_balance = initial_balance  # Highest account balance observed

        # Close SQL connections
        self.backtest_sqlhelper.close_connection()
        data_sqlhelper.close_connection()
        self.backtest_sqlhelper = None
        data_sqlhelper = None

    # def __repr__(self):
    #     return f"Backtester({self.strategy.NAME}, {self.forex_pair}, {self.timeframe}) | {self.strategy.PARAMETER_SETTINGS}"

    def _calculate_metrics(self):
        total_trades = len(self.all_trades)

        # Set up the initial metrics dictionary, including new metrics.
        metrics = {
            "Data_Start_Date": self.data_start_date,
            "Data_End_Date": self.data_end_date,
            "Trading_Start_Date": self.data["Timestamp"].iloc[self.trading_start_index],
            "Total_Trades": total_trades,
            "Winning_Trades": 0,
            "Gross_Profit": 0.0,
            "Gross_Loss": 0.0,
            "Net_Profit": 0.0,
            "Total_Return_Pct": 0.0,
            "Win_Rate": 0.0,
            "Profit_Factor": 0.0,
            "Max_Drawdown": getattr(self, "max_drawdown", 0.0),
            "Max_Drawdown_Pct": getattr(self, "max_drawdown_pct", 0.0),
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
            1 for trade in self.all_trades if trade.pnl is not None and trade.pnl > 0
        )
        gross_profit = sum(
            trade.pnl
            for trade in self.all_trades
            if trade.pnl is not None and trade.pnl > 0
        )
        gross_loss = sum(
            trade.pnl
            for trade in self.all_trades
            if trade.pnl is not None and trade.pnl < 0
        )
        initial_balance = self.initial_balance
        final_balance = self.all_trades[-1].balance_after_trade
        net_profit = final_balance - initial_balance
        total_return_pct = (
            (net_profit / initial_balance) * 100 if initial_balance != 0 else 0
        )
        win_rate = winning_trades / total_trades
        profit_factor = (
            (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
        )
        average_trade_duration = (
            sum(
                trade.duration
                for trade in self.all_trades
                if trade.duration is not None
            )
            / total_trades
        )

        # Compute trade returns (as percentages) for Sharpe, Sortino, and Trade Expectancy.
        trade_returns = [
            trade.return_pct
            for trade in self.all_trades
            if trade.return_pct is not None
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
            (total_return_pct / self.max_drawdown_pct)
            if self.max_drawdown_pct != 0
            else float("inf")
        )

        recovery_factor = (
            (net_profit / self.max_drawdown) if self.max_drawdown != 0 else float("inf")
        )

        # Win/Loss Ratio (Average): based on percentage returns.
        wins_pct = [
            trade.return_pct
            for trade in self.all_trades
            if trade.return_pct is not None and trade.return_pct > 0
        ]
        losses_pct = [
            trade.return_pct
            for trade in self.all_trades
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
        for trade in self.all_trades:
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
                for trade in self.all_trades
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
        metrics["Max_Drawdown"] = round(self.max_drawdown, 2)
        metrics["Max_Drawdown_Pct"] = round(self.max_drawdown_pct, 2)
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

    def run_backtest(self):
        """
        Executes the backtest using the provided strategy.
        Now uses generate_trade_plan() from Strategy to determine all trade logic.
        """
        closes = self.data["Close"].values
        timestamps = self.data["Timestamp"].values
        quote_to_usd_rates = (
            self.quote_to_usd_data["Close"].values
            if self.quote_to_usd_data is not None
            else None
        )

        for index, row in enumerate(self.data.itertuples(index=False)):

            # Skip the first 100 rows so we have a semblance of historical data
            if index < self.trading_start_index:
                continue

            current_price = closes[index]
            timestamp = timestamps[index]

            quote_to_usd_rate = None
            if quote_to_usd_rates is not None:
                quote_to_usd_rate = quote_to_usd_rates[index]

            # Generate zero or more trade plans (entries and/or exits) using strategy logic
            plans: list[strategies.TradePlan] = self.strategy.generate_trade_plan(
                # Slice precomputed dataset up to current index to mimic real-time data
                index,
                self.position,
                self.balance,
                quote_to_usd_rate,
            )

            # --- Step 1: Check SL/TP exit for open trades BEFORE executing any strategy ENTRY/EXIT ---
            exited_trades = []
            for trade in self.open_trades:
                # Update PositionManager with current price
                # This updates its internal stop loss based on strategy logic,
                # then returns a tuple of (exit_condition, exit_price) or (None, None) if no exit
                exit_condition, exit_price, event = trade.plan.position_manager.update(
                    current_price
                )
                if event:
                    trade.position_events.append((timestamp, event))

                # Close trade if exit condition is met (STOP_LOSS or TAKE_PROFIT)
                if exit_condition:
                    trade.position_events.append((timestamp, exit_condition))
                    trade.close_trade(
                        timestamp,
                        exit_price,
                        self.commission_per_lot,
                        quote_to_usd_rate,
                    )
                    self.balance = trade.balance_after_trade
                    exited_trades.append(trade)

            # Remove only trades that exited
            self.open_trades = [t for t in self.open_trades if t not in exited_trades]

            # --- Step 2: Execute new trade plans (entries + strategy EXITs)
            for plan in plans:
                # Exit logic (from EXIT StrategyPlan) — if we're holding a position
                if plan.tag == "EXIT" and self.open_trades:
                    exit_price = (
                        current_price * (1 - self.slippage)
                        if self.position == 1
                        else current_price * (1 + self.slippage)
                    )
                    for trade in self.open_trades:
                        trade.close_trade(
                            timestamp,
                            exit_price,
                            self.commission_per_lot,
                            quote_to_usd_rate,
                        )
                        self.balance = trade.balance_after_trade
                    self.open_trades = []

                # Entry logic — only run when flat
                elif self.position == 0 and not self.open_trades:
                    # Skip if any critical values are missing or invalid
                    if any(
                        math.isnan(x) for x in [plan.entry_price, plan.stop_loss]
                    ) or plan.direction not in ["BUY", "SELL"]:
                        continue

                    entry_price = (
                        plan.entry_price * (1 + self.slippage)
                        if plan.direction == "BUY"
                        else plan.entry_price * (1 - self.slippage)
                    )

                    # Ensure we have enough margin to open this trade
                    margin_required = (
                        strategies.PositionCalculator.calculate_required_margin(
                            self.forex_pair,
                            plan.units,
                            self.leverage,
                            entry_price,
                            quote_to_usd_rate,
                        )
                    )
                    if margin_required > self.margin_remaining:
                        continue  # Skip this plan if insufficient margin

                    # Estimate pip value
                    pip_value = (
                        strategies.PositionCalculator.calculate_current_pip_value(
                            self.forex_pair, plan.units, entry_price, quote_to_usd_rate
                        )
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

            # Track equity high to compute drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

            # Compute drawdown
            drawdown = self.peak_balance - self.balance
            drawdown_pct = (
                (drawdown / self.peak_balance) * 100 if self.peak_balance > 0 else 0
            )
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.max_drawdown_pct = drawdown_pct

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

    def calculate_composite_score(self):
        """
        Calculates a composite score for evaluating strategy performance,
        incorporating performance, risk, consistency, and activity metrics.
        """

        # --- Core performance metrics ---
        expectancy_per_day = self.get_metric(
            "Expectancy_Per_Day_Pct"
        )  # Daily expected return (%)
        trade_expectancy = self.get_metric(
            "Trade_Expectancy_Pct"
        )  # Per-trade return expectation (%)
        profit_factor = self.get_metric(
            "Profit_Factor"
        )  # Ratio of gross profit to gross loss
        win_loss_ratio = self.get_metric("Win_Loss_Ratio")  # Avg win % / avg loss %
        trades_per_day = self.get_metric(
            "Trades_Per_Day"
        )  # Average trades per calendar day

        # --- Base score weighting ---
        score = (
            expectancy_per_day * 1.5  # Primary performance metric
            + trade_expectancy * 1.0  # Efficiency per trade
            + profit_factor * 0.5  # Risk-adjusted reward
            + win_loss_ratio * 0.5  # Reward vs. risk skew
            + trades_per_day * 0.25  # Encourages some activity
        )

        # --- Pull supporting risk/volatility metrics ---
        max_drawdown_pct = self.get_metric(
            "Max_Drawdown"
        )  # Max % drop from equity peak
        trade_std = self.get_metric("Trade_Return_Std")  # Volatility of returns
        sharpe_ratio = self.get_metric("Sharpe_Ratio")  # Return / total volatility
        sortino_ratio = self.get_metric("Sortino_Ratio")  # Return / downside volatility
        max_margin_required_pct = self.get_metric("Max_Margin_Required_Pct")
        timeframe = self.timeframe

        # --- Normalize trade activity by timeframe expectations ---
        trades_per_day_threshold = config.MIN_TRADES_PER_DAY[
            timeframe
        ]  # Adjusted baseline
        activity_penalty = min(1.0, trades_per_day / trades_per_day_threshold)
        score *= (
            activity_penalty  # Penalize for very low activity relative to timeframe
        )

        # --- Penalize high drawdown (exponential) ---
        # At 20% drawdown → 50% penalty, at 40% → 80%, at 60% → 90%
        score *= 1 / (1 + (max_drawdown_pct / 20) ** 2)

        # --- Penalize high volatility of returns ---
        # A strategy with 0% std gets 1.0, 20% std gets 0.5, 40% gets ~0.33
        score *= 1 / (1 + trade_std / 20)

        # --- Bonus for strong Sharpe/Sortino ratios ---
        # Cap bonuses so they don’t dominate score
        if sharpe_ratio > 1.0:
            score += min((sharpe_ratio - 1.0), 3.0) * 0.25
        if sortino_ratio > 2.0:
            score += min((sortino_ratio - 2.0), 3.0) * 0.25

        # --- Penalize high leverage / aggressive sizing ---
        if max_margin_required_pct > 80:
            score *= 0.85  # Flat penalty for risk exposure

        # --- Clamp final score to a max (optional, avoids runaway inflation) ---
        # score = min(score, 25.0)

        return score


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
