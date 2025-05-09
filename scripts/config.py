import os

DATA_FOLDER = "data"
BACKTESTING_FOLDER = "backtesting"
BACKTESTING_DB_NAME = "Backtesting.db"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(BACKTESTING_FOLDER, exist_ok=True)

MAJOR_FOREX_PAIRS = [
    "EUR/USD",
    "USD/JPY",
    "GBP/USD",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD",
    "USD/SGD",
    "EUR/GBP",
    "EUR/JPY",
    "EUR/CHF",
    "EUR/AUD",
    "EUR/CAD",
    "EUR/NZD",
    "GBP/JPY",
    "GBP/CHF",
    "GBP/AUD",
    "GBP/CAD",
    "GBP/NZD",
    "AUD/JPY",
    "AUD/CHF",
    "AUD/CAD",
    "AUD/NZD",
    "CAD/JPY",
    "CAD/CHF",
    "NZD/JPY",
    "NZD/CHF",
    "CHF/JPY",
    "EUR/SGD",
    "USD/SGD",
    "GBP/SGD",
    "AUD/SGD",
    "SGD/JPY",
    "NZD/SGD",
    "CAD/SGD",
    "CHF/SGD",
]

TIMEFRAMES = [
    # "5_minute",
    # "15_minute",
    # "30_minute",
    # "1_hour",
    # "2_hour",
    # "4_hour",
    "1_day",
]

TIMESPAN_MULTIPLIER_PAIRS = [
    (5, "minute"),
    (15, "minute"),
    (30, "minute"),
    (1, "hour"),
    (2, "hour"),
    (4, "hour"),
    (1, "day"),
]

# === RISK MANAGEMENT CONSTANTS ===
DEFAULT_ATR_SL_MULTIPLIER = 1.5
DEFAULT_TP_MULTIPLIER = 1.0
RISK_PER_TRADE = 0.01

# Trade Entry and Exit Signals
ENTER_LONG = "ENTER_LONG"
EXIT_LONG = "EXIT_LONG"
ENTER_SHORT = "ENTER_SHORT"
EXIT_SHORT = "EXIT_SHORT"
NO_SIGNAL = "NO_SIGNAL"

# === INDICATOR FUNCTION SIGNALS ===
# Trend Signals
BULLISH_TREND = "BULLISH_TREND"
BEARISH_TREND = "BEARISH_TREND"
NEUTRAL_TREND = "NEUTRAL_TREND"

# Crossover and Price Relation Signals
BULLISH_SIGNAL = "BULLISH_CROSSOVER"
BEARISH_SIGNAL = "BEARISH_CROSSOVER"

# Overbought/Oversold Conditions
OVERBOUGHT = "OVERBOUGHT"
OVERSOLD = "OVERSOLD"

# Divergence Signals
BULLISH_DIVERGENCE = "BULLISH_DIVERGENCE"
BEARISH_DIVERGENCE = "BEARISH_DIVERGENCE"

# Volume Signals
HIGH_VOLUME = "HIGH_VOLUME"
LOW_VOLUME = "LOW_VOLUME"

# Volatility Signals
HIGH_VOLATILITY = "HIGH_VOLATILITY"
LOW_VOLATILITY = "LOW_VOLATILITY"
INCREASING_VOLATILITY = "INCREASING_VOLATILITY"
DECREASING_VOLATILITY = "DECREASING_VOLATILITY"
STABLE_VOLATILITY = "STABLE_VOLATILITY"

# Miscellaneous Signals
INCONCLUSIVE = "INCONCLUSIVE"

# === BACKTESTING CONSTANTS ===
# Baseline number of trades per day (for calculating composite score)
MIN_TRADES_PER_DAY = {
    "5_minute": 5.0,
    "15_minute": 3.0,
    "30_minute": 2.0,
    "1_hour": 1.0,
    "2_hour": 0.5,
    "4_hour": 0.25,
    "1_day": 0.03,  # 1 trade every ~30 days
}
