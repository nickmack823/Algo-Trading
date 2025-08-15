import os

import pandas as pd

# Directories
DATA_FOLDER = "data"
BACKTESTING_FOLDER = "backtesting"
BACKTESTING_DB_NAME = "Backtesting.db"
OPTUNA_STUDIES_FOLDER = "backtesting/optuna_studies"
OPTUNA_REPORTS_FOLDER = "backtesting/optuna_reports"
TEMP_CACHE_FOLDER = "temp"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(BACKTESTING_FOLDER, exist_ok=True)
os.makedirs(OPTUNA_STUDIES_FOLDER, exist_ok=True)
os.makedirs(OPTUNA_REPORTS_FOLDER, exist_ok=True)
os.makedirs(TEMP_CACHE_FOLDER, exist_ok=True)

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
    "5_minute",
    "15_minute",
    "30_minute",
    "1_hour",
    "2_hour",
    "4_hour",
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
# Data start/end dates
START_DATE = "2022-05-08"
END_DATE = "2025-05-07"  # one day before last in data
TIMEFRAME_DATE_RANGES_PHASE_1_AND_2 = {
    "1_day": {
        "from_date": START_DATE,  # full 3 years
        "to_date": END_DATE,
    },
    "4_hour": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=365)).strftime(
            "%Y-%m-%d"
        ),  # last 1 year
        "to_date": END_DATE,
    },
    "2_hour": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=182)).strftime(
            "%Y-%m-%d"
        ),  # last 0.5 years
        "to_date": END_DATE,
    },
    "1_hour": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=120)).strftime(
            "%Y-%m-%d"
        ),  # last 4 months
        "to_date": END_DATE,
    },
    "30_minute": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=90)).strftime(
            "%Y-%m-%d"
        ),  # last 3 months
        "to_date": END_DATE,
    },
    "15_minute": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=60)).strftime(
            "%Y-%m-%d"
        ),  # last 2 months
        "to_date": END_DATE,
    },
    "5_minute": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=30)).strftime(
            "%Y-%m-%d"
        ),  # last 1 months
        "to_date": END_DATE,
    },
}
TIMEFRAME_DATE_RANGES_PHASE3 = {
    "1_day": {
        "from_date": START_DATE,  # full 3 years
        "to_date": END_DATE,
    },
    "4_hour": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=730)).strftime(
            "%Y-%m-%d"
        ),  # last 2 years
        "to_date": END_DATE,
    },
    "2_hour": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=547)).strftime(
            "%Y-%m-%d"
        ),  # last 1.5 years
        "to_date": END_DATE,
    },
    "1_hour": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=365)).strftime(
            "%Y-%m-%d"
        ),  # last 1 year
        "to_date": END_DATE,
    },
    "30_minute": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=270)).strftime(
            "%Y-%m-%d"
        ),  # last 9 months
        "to_date": END_DATE,
    },
    "15_minute": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=180)).strftime(
            "%Y-%m-%d"
        ),  # last 6 months
        "to_date": END_DATE,
    },
    "5_minute": {
        "from_date": (pd.to_datetime(END_DATE) - pd.Timedelta(days=90)).strftime(
            "%Y-%m-%d"
        ),  # last 3 months
        "to_date": END_DATE,
    },
}

PRUNE_THRESHOLD_FACTOR = 0.2  # Prune if strategy executes < 20% of expected trades
# Baseline number of trades per day (for calculating composite score)
MIN_TRADES_PER_DAY = {
    "5_minute": 3.0,  # ~1 trade every 1.5 hours
    "15_minute": 2.0,  # ~1 trade every 2 hours
    "30_minute": 1.5,  # ~1 trade every 4 hours
    "1_hour": 0.5,  # ~1 trade every 2 days
    "2_hour": 0.2,  # ~1 trade every 5 days (~70/year)
    "4_hour": 0.1,  # ~1 trade every 10 days (~35/year)
    "1_day": 0.05,  # ~1 trade every 20 days (~18/year)
}
# Number of optuna trials per timeframe
# TRIALS_BY_TIMEFRAME = {
#     "1_day": 500,
#     "4_hour": 450,
#     "2_hour": 400,
#     "1_hour": 150,
#     "30_minute": 120,
#     "15_minute": 100,
#     "5_minute": 75,
# }

# Maximum number of optuna trials
TRIALS_UPPER_BOUND = 500
# Number of trials to allow without score improvement before early stopping
# IMPROVEMENT_CUTOFF_BY_TIMEFRAME = {
#     "1_day": 750,  # 50,
#     "4_hour": 750,  # 40,
#     "2_hour": 750,  # 30,
#     "1_hour": 20,
#     "30_minute": 15,
#     "15_minute": 10,
#     "5_minute": 8,
# }

# === Phases for final tuning ===
N_STARTUP_TRIALS_PERCENTAGE = 0.15
PHASE2_TOP_PERCENT = 10


# === OANDA Config ===
OANDA_API_KEY = "9dc7d56d2f3c584dcd04947f2c773983-f3c6f66bb43e9f533c910a636908995f"
OANDA_ACCOUNT_ID = "101-001-9539917-002"

MY_LOCAL_TIMEZONE = "America/New_York"
