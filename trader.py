# oanda_api.py
import json

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import pandas as pd
import pytz

from scripts import strategies, utilities
from scripts.config import OANDA_ACCOUNT_ID, OANDA_API_KEY
from scripts.indicators import find_indicator_config

timeframe_to_granularity = {
    "1_day": "D",
    "4_hour": "H4",
    "2_hour": "H2",
}


class OandaAPI:
    def __init__(self):
        self.client = oandapyV20.API(access_token=OANDA_API_KEY)
        self.account_id = OANDA_ACCOUNT_ID

    def get_candles(
        self, pair="EUR_USD", granularity="H1", count=100, local_tz="America/New_York"
    ):
        """
        Fetches candle data, converts UTC to local time, returns a formatted DataFrame.

        Args:
            pair (str): Trading pair like "EUR_USD"
            granularity (str): Candle interval (e.g. "H1", "M15")
            count (int): Number of candles to retrieve
            local_tz (str): IANA timezone string for conversion (e.g. "America/New_York")

        Returns:
            pd.DataFrame: Formatted candles with local time
        """

        params = {"granularity": granularity, "count": count, "price": "M"}

        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        self.client.request(r)
        candles = r.response["candles"]

        tz = pytz.timezone(local_tz)
        data = []
        for c in candles:
            if c["complete"]:
                utc_dt = pd.to_datetime(c["time"]).tz_convert("UTC")
                local_dt = utc_dt.tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")

                ohlc = c["mid"]
                data.append(
                    {
                        "Time": local_dt,
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
        return float(r.response["account"]["balance"])

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


def run_strategy_once(oanda: OandaAPI, strategy: strategies.NNFXStrategy):
    """
    Pulls data, generates trade plan from strategy, and executes orders accordingly.
    Uses real account balance and quote-to-USD rate for sizing.
    """
    pair = strategy.FOREX_PAIR[:3] + "_" + strategy.FOREX_PAIR[3:]
    timeframe = strategy.TIMEFRAME
    granularity = timeframe_to_granularity[timeframe]

    # 1. Get latest candle data
    df = oanda.get_candles(pair=pair, granularity=granularity, count=300)
    df = df.rename(columns={"Time": "Timestamp"})

    # 2. Prepare data
    strategy.prepare_data(df, use_cache=False)
    current_index = len(df) - 1
    current_position = 0  # TODO: Replace with actual live trade state

    # 3. Use live balance and quote-to-USD rate
    balance = oanda.get_account_balance()
    quote_to_usd = oanda.get_quote_to_usd_rate(pair)

    # 4. Get TradePlan(s)
    trade_plans = strategy.generate_trade_plan(
        current_index=current_index,
        current_position=current_position,
        balance=balance,
        quote_to_usd_rate=quote_to_usd,
    )

    for trade in trade_plans:
        if trade.tag == "EXIT":
            print(f"[EXIT] signal: {trade.direction} at {trade.entry_price}")
            # You may want to implement `close_trade` here with matching direction
            continue

        sl_distance = abs(trade.entry_price - trade.stop_loss)
        tp_distance = (
            abs(trade.entry_price - trade.take_profit) if trade.take_profit else None
        )

        print(
            f"[ENTRY] {trade.direction} | Entry: {trade.entry_price}, SL: {sl_distance:.5f}, TP: {tp_distance}"
        )

        oanda.place_market_order(
            pair=pair,
            units=trade.units if trade.direction == "BUY" else -trade.units,
            sl_pips=sl_distance,
            tp_pips=tp_distance,
        )


if __name__ == "__main__":
    api = OandaAPI()

    # Load best_strategies.json
    best_strategies = []
    with open("best_strategies.json", "r") as f:
        best_strategies = json.load(f)

    test_strategy = utilities.load_strategy_from_dict(best_strategies[0])

    run_strategy_once(api, test_strategy)
