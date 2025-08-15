import json
import logging
import time
import winsound
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import pandas as pd
import pyttsx3
import pytz
from oandapyV20.endpoints.positions import OpenPositions

from scripts import strategies
from scripts.config import MY_LOCAL_TIMEZONE, OANDA_ACCOUNT_ID, OANDA_API_KEY
from scripts.sql import OandaSQLHelper
from scripts.utilities import load_strategy_from_dict, seconds_to_dhms_str

pytts_engine = pyttsx3.init()

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)

# Suppress oandapyV20 logging
logging.getLogger("oandapyV20").setLevel(logging.WARNING)

timeframe_to_granularity = {
    "1_day": "D",
    "4_hour": "H4",
    "2_hour": "H2",
}


def speak(text: str):
    pytts_engine.say(text)
    pytts_engine.runAndWait()


def get_current_local_datetime() -> datetime:
    return datetime.now(pytz.timezone(MY_LOCAL_TIMEZONE))


@dataclass
class TradeAction:
    type: Literal["ENTRY", "EXIT", "ERROR", "WARNING", "CANDLE"]
    direction: Optional[str] = None
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    units: Optional[float] = None
    risk_pct: Optional[float] = None
    source: Optional[str] = None
    strategy: Optional[str] = None
    tag: Optional[str] = None
    note: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RunningStrategy:
    strategy: strategies.NNFXStrategy
    strategy_id: int
    pair: str
    timeframe: str
    granularity: str
    last_candle_time: str
    current_position: int


def get_next_candle_close(granularity: str, current_time: datetime) -> datetime:
    """Returns the next candle close time based on granularity."""
    # Round down current time to nearest granularity
    if granularity == "M1":
        interval = timedelta(minutes=1)
    elif granularity == "M5":
        interval = timedelta(minutes=5)
    elif granularity == "M15":
        interval = timedelta(minutes=15)
    elif granularity == "M30":
        interval = timedelta(minutes=30)
    elif granularity == "H1":
        interval = timedelta(hours=1)
    elif granularity == "H2":
        interval = timedelta(hours=2)
    elif granularity == "H4":
        interval = timedelta(hours=4)
    elif granularity == "D":
        interval = timedelta(days=1)
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

    remainder = (
        current_time - datetime.min.replace(tzinfo=current_time.tzinfo)
    ) % interval
    return current_time + (interval - remainder)


class OandaAPI:
    def __init__(self):
        environment = "practice" if TRADING_MODE == "demo" else "live"
        self.client = oandapyV20.API(
            access_token=OANDA_API_KEY, environment=environment
        )
        self.account_id = OANDA_ACCOUNT_ID

    def get_candles(self, pair="EUR_USD", granularity="H1", count=100):
        """
        Fetches candle data, converts UTC to local time, returns a formatted DataFrame.

        Args:
            pair (str): Trading pair like "EUR_USD"
            granularity (str): Candle interval (e.g. "H1", "M15")
            count (int): Number of candles to retrieve

        Returns:
            pd.DataFrame: Formatted candles with local time
        """

        params = {"granularity": granularity, "count": count, "price": "M"}

        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        self.client.request(r)
        candles = r.response["candles"]

        tz = pytz.timezone(MY_LOCAL_TIMEZONE)
        data = []
        for c in candles:
            if c["complete"]:
                utc_dt = pd.to_datetime(c["time"]).tz_convert("UTC")
                local_dt = utc_dt.tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")

                ohlc = c["mid"]
                data.append(
                    {
                        "Timestamp": local_dt,
                        "Open": float(ohlc["o"]),
                        "High": float(ohlc["h"]),
                        "Low": float(ohlc["l"]),
                        "Close": float(ohlc["c"]),
                        "Volume": int(c["volume"]),
                    }
                )

        return pd.DataFrame(data)

    def get_open_trades(self):
        r = trades.OpenTrades(self.account_id)
        self.client.request(r)
        return r.response.get("trades", [])

    def get_account_balance(self) -> float:
        r = accounts.AccountSummary(accountID=self.account_id)
        self.client.request(r)
        return round(float(r.response["account"]["balance"]), 2)

    def get_price(self, pair: str) -> dict:
        params = {"instruments": pair}
        r = pricing.PricingInfo(accountID=self.account_id, params=params)
        self.client.request(r)
        bid = float(r.response["prices"][0]["bids"][0]["price"])
        ask = float(r.response["prices"][0]["asks"][0]["price"])
        return {"mid": (bid + ask) / 2, "bid": bid, "ask": ask}

    def get_quote_to_usd_rate(self, pair: str) -> float:
        """
        Tries to get the exchange rate from quote currency to USD.
        For USD quote pairs (e.g., EUR/USD), return 1.
        For others (e.g., EUR/GBP), get price for quote/USD (e.g., GBP/USD).
        """
        quote_currency = pair[-3:]
        if quote_currency == "USD":
            return 1.0

        reverse_pair = f"{quote_currency}_USD"
        try:
            return self.get_price(reverse_pair)["mid"]
        except:
            # If quote/USD is invalid, try USD/quote and invert it
            reverse_pair = f"USD_{quote_currency}"
            try:
                price = self.get_price(reverse_pair)["mid"]
                return 1 / price
            except:
                raise ValueError(
                    f"Could not resolve quote-to-USD rate for: {quote_currency}"
                )

    def close_trade(self, trade_id):
        r = trades.TradeClose(accountID=self.account_id, tradeID=trade_id)
        self.client.request(r)
        return r.response

    def place_market_order(self, pair, units, sl_pips=None, tp_pips=None):
        """
        Places a market order with optional stop loss and take profit.

        Args:
            pair (str): Instrument to trade (e.g. "EUR_USD")
            units (int): Number of units to trade. Positive = Buy, Negative = Sell
            sl_pips (float): Optional. Stop loss distance in price units (not pips)
            tp_pips (float): Optional. Take profit distance in price units (not pips)

        Returns:
            dict: API response from OANDA
        """

        # The order payload
        order_data = {
            "order": {
                "instrument": pair,  # The forex pair or CFD symbol
                "units": str(units),  # Must be string; positive = buy, negative = sell
                "type": "MARKET",  # Market order type; executes immediately
                "positionFill": "DEFAULT",  # Options:
                # "DEFAULT" - respects account settings
                # "OPEN_ONLY" - only opens a new position
                # "REDUCE_FIRST" - reduces existing opposite position
                # "REDUCE_ONLY" - only reduces existing position
            }
        }

        # Optional SL/TP logic using 'distance' from entry price.
        # OANDA also allows:
        # - stopLossOnFill = { "price": "1.09500" }  # Absolute price
        # - trailingStopLossOnFill = { "distance": "0.0020" }
        if sl_pips:
            order_data["order"]["stopLossOnFill"] = {"distance": f"{sl_pips:.5f}"}

        if tp_pips:
            order_data["order"]["takeProfitOnFill"] = {"distance": f"{tp_pips:.5f}"}

        # Other optional fields you can add later:
        # - "clientExtensions": { "id": "custom_trade_id", "comment": "strategy signal" }
        # - "timeInForce": "FOK" or "IOC" or "GTC"  # Execution rule (fill or kill, etc.)
        # - "tradeClientExtensions": {}  # Custom metadata for the resulting trade

        # Execute the order
        r = orders.OrderCreate(self.account_id, data=order_data)
        self.client.request(r)

        return r.response

    def get_open_position_summary(self) -> str:
        try:
            r = OpenPositions(accountID=self.account_id)
            self.client.request(r)
            positions = r.response.get("positions", [])

            summaries = []
            for pos in positions:
                instrument = pos["instrument"]
                long_units = int(pos["long"]["units"])
                short_units = int(pos["short"]["units"])

                if long_units > 0:
                    summaries.append(f"{instrument}: Long {long_units}")
                if short_units > 0:
                    summaries.append(f"{instrument}: Short {abs(short_units)}")

            return " | ".join(summaries) if summaries else "None"
        except Exception as e:
            return f"Error fetching positions: {e}"


def run_strategy(
    oanda: OandaAPI,
    strategy: strategies.NNFXStrategy,
    current_position: int,
    balance: float,
    quote_to_usd_rate: float,
) -> list[TradeAction]:
    actions = []

    # Step 1: Prepare data with indicators already attached
    if strategy.data_with_indicators is None:
        raise ValueError(
            "strategy.prepare_data() must be called before run_strategy_once"
        )

    # Step 2: Get current index (last row)
    current_index = len(strategy.data_with_indicators) - 1

    # Step 3: Generate trade plans
    trade_plans: list[strategies.TradePlan] = strategy.generate_trade_plan(
        current_index=current_index,
        current_position=current_position,
        balance=balance,
        quote_to_usd_rate=quote_to_usd_rate,
    )

    # Step 4: Execute trade plans
    for plan in trade_plans:
        action = TradeAction(
            type=plan.tag,
            direction=plan.direction,
            price=plan.entry_price,
            sl=plan.stop_loss,
            tp=plan.take_profit,
            units=plan.units,
            risk_pct=plan.risk_pct,
            source=plan.source,
            strategy=plan.strategy,
            tag=plan.tag,
        )

        if plan.tag == "EXIT":
            logging.info(f"[EXIT] {strategy.FOREX_PAIR} - Closing {plan.direction}")
            actions.append(action)
            # (Optional: track and close open trades in memory/simulation)

        elif plan.tag == "ENTRY":
            logging.info(
                f"[ENTRY] {strategy.FOREX_PAIR} - {plan.direction} at {plan.entry_price}"
            )

            try:
                oanda.place_market_order(
                    pair=strategy.FOREX_PAIR,
                    units=plan.units if plan.direction == "BUY" else -plan.units,
                    sl_pips=(
                        abs(plan.entry_price - plan.stop_loss)
                        if plan.stop_loss
                        else None
                    ),
                    tp_pips=(
                        abs(plan.entry_price - plan.take_profit)
                        if plan.take_profit
                        else None
                    ),
                )
            except Exception as e:
                action.type = "ERROR"
                action.error = str(e)
                logging.error(f"[ERROR] Order placement failed: {e}")

            actions.append(action)

        else:
            logging.warning(f"[WARNING] {strategy.FOREX_PAIR} - Unknown tag {plan.tag}")
            actions.append(
                TradeAction(
                    type="WARNING", note=f"Unhandled tag: {plan.tag}", tag=plan.tag
                )
            )

    return actions


def main_loop(n_strategies):
    api = OandaAPI()
    db = OandaSQLHelper()

    with open("best_strategies.json", "r") as f:
        strat_configs = json.load(f)

    running_strategies = []

    for config in strat_configs[:n_strategies]:  # Top N strategies
        pair = config["pair"]
        timeframe = config["timeframe"]

        logging.info(f"Loading strategy for {pair} {timeframe}")

        strategy = load_strategy_from_dict(config)

        live_strategy_id = db.upsert_strategy(
            name="NNFX",
            pair=pair,
            timeframe=timeframe,
            parameters=config["indicators"],
            mode=TRADING_MODE,
        )

        running_strategies.append(
            RunningStrategy(
                strategy=strategy,
                strategy_id=live_strategy_id,
                pair=pair,
                timeframe=timeframe,
                granularity=timeframe_to_granularity[timeframe],
                last_candle_time=None,
                current_position=0,
            )
        )

    while True:

        winsound.PlaySound("bicycle_bell_quiet.wav", winsound.SND_ASYNC)

        for running_strategy in running_strategies:

            strategy: strategies.NNFXStrategy = running_strategy.strategy
            live_strategy_id = running_strategy.strategy_id
            pair = running_strategy.pair
            granularity = running_strategy.granularity
            timeframe = running_strategy.timeframe
            last_candle_time = running_strategy.last_candle_time

            try:

                candles = api.get_candles(
                    pair=pair.replace("/", "_"), granularity=granularity, count=200
                )
                latest_time = candles.iloc[-1]["Timestamp"]

                if last_candle_time != latest_time:
                    logging.info(f"[{pair}] New candle found {latest_time}")
                    # db.log_trade_action(
                    #     strategy_id, pair, timeframe, act, mode=TRADING_MODE
                    # )

                    balance = api.get_account_balance()
                    quote_to_usd = api.get_quote_to_usd_rate(pair.replace("/", ""))
                    strategy.prepare_data(candles, use_cache=False)

                    actions_taken = run_strategy(
                        api,
                        strategy,
                        current_position=running_strategy.current_position,
                        balance=balance,
                        quote_to_usd_rate=quote_to_usd,
                    )

                    if len(actions_taken) == 0:
                        logging.info(f"[{pair}] No actions taken")

                    for act in actions_taken:
                        # Log action
                        db.log_trade_action(
                            live_strategy_id, pair, timeframe, act, mode=TRADING_MODE
                        )

                        # Update position tracking for running strategy
                        if act.type == "ENTRY":
                            running_strategy.current_position = (
                                1 if act.direction == "BUY" else -1
                            )
                        elif act.type == "EXIT":
                            running_strategy.current_position = 0

                        speak(f"{act.type} {act.direction or ''} on {pair}")

                    running_strategy.last_candle_time = latest_time

            except Exception as e:
                error_action = TradeAction(
                    type="ERROR",
                    note="Exception in strategy execution",
                    error=str(e),
                )
                db.log_trade_action(
                    live_strategy_id, pair, timeframe, error_action, mode=TRADING_MODE
                )

        # After processing all strategies, calculate when to wake up next
        now = get_current_local_datetime()
        next_closes = []

        for running_strategy in running_strategies:
            granularity = running_strategy.granularity
            next_close = get_next_candle_close(granularity, now)
            next_closes.append(next_close)

        # Wake up 1 minute after the earliest next close
        target_time = min(next_closes) + timedelta(minutes=1)
        sleep_duration = (target_time - get_current_local_datetime()).total_seconds()
        sleep_duration = max(sleep_duration, 15)  # Don't sleep too little

        logging.info(
            f"""Sleeping {seconds_to_dhms_str(sleep_duration)} until 1 minute after next candle close...
            \nCurrent Balance: {api.get_account_balance()}
            \nOpen Positions: {api.get_open_position_summary()}\n"""
        )

        remaining = int(sleep_duration)
        try:
            while remaining > 0:
                print(
                    f"\r⏳ Next candle in: {seconds_to_dhms_str(remaining)}",
                    end="",
                    flush=True,
                )
                time.sleep(1)
                remaining -= 1
        except KeyboardInterrupt:
            print("\nInterrupted manually.")


if __name__ == "__main__":
    TRADING_MODE = "demo"
    N_CONCURRENT_STRATEGIES = 200

    main_loop(N_CONCURRENT_STRATEGIES)
