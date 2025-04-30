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

BASIC_COLUMNS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "VWAP",
    "Transactions",
    "ATR",
]

# Indicator columns
FINAL_INDICATOR_COLUMNS = [
    "AD",
    "ADOSC",
    "ADX",
    "ADXR",
    "APO",
    "AROONOSC",
    "AROON_down",
    "AROON_up",
    "ATR",
    "AVGPRICE",
    "BETA",
    "BOP",
    "Bollinger_lower",
    "Bollinger_middle",
    "Bollinger_upper",
    "CANDLE_2CROWS",
    "CANDLE_3BLACKCROWS",
    "CANDLE_3INSIDE",
    "CANDLE_3LINESTRIKE",
    "CANDLE_3STARSINSOUTH",
    "CANDLE_3WHITESOLDIERS",
    "CANDLE_ABANDONEDBABY",
    "CANDLE_ADVANCEBLOCK",
    "CANDLE_BELTHOLD",
    "CANDLE_BODYPERCENT",
    "CANDLE_BREAKAWAY",
    "CANDLE_CLOSINGMARUBOZU",
    "CANDLE_CONCEALBABYSWALL",
    "CANDLE_COUNTERATTACK",
    "CANDLE_DARKCLOUDCOVER",
    "CANDLE_DOJI",
    "CANDLE_DOJISTAR",
    "CANDLE_DRAGONFLYDOJI",
    "CANDLE_ENGULFING",
    "CANDLE_EVENINGDOJISTAR",
    "CANDLE_EVENINGSTAR",
    "CANDLE_GAPSIDESIDEWHITE",
    "CANDLE_GRAVESTONEDOJI",
    "CANDLE_HAMMER",
    "CANDLE_HANGINGMAN",
    "CANDLE_HARAMI",
    "CANDLE_HARAMICROSS",
    "CANDLE_HIGHWAVE",
    "CANDLE_HIKKAKE",
    "CANDLE_HIKKAKEMOD",
    "CANDLE_HOMINGPIGEON",
    "CANDLE_IDENTICAL3CROWS",
    "CANDLE_INNECK",
    "CANDLE_INVERTEDHAMMER",
    "CANDLE_KICKING",
    "CANDLE_KICKINGBYLENGTH",
    "CANDLE_LADDERBOTTOM",
    "CANDLE_LONGLEGGEDDOJI",
    "CANDLE_LONGLINE",
    "CANDLE_MARUBOZU",
    "CANDLE_MATCHINGLOW",
    "CANDLE_MATHOLD",
    "CANDLE_MORNINGDOJISTAR",
    "CANDLE_MORNINGSTAR",
    "CANDLE_ONNECK",
    "CANDLE_PIERCING",
    "CANDLE_RANGE",
    "CANDLE_RICKSHAWMAN",
    "CANDLE_RISEFALL3METHODS",
    "CANDLE_SEPARATINGLINES",
    "CANDLE_SHOOTINGSTAR",
    "CANDLE_SHORTLINE",
    "CANDLE_SPINNINGTOP",
    "CANDLE_STALLEDPATTERN",
    "CANDLE_STICKSANDWICH",
    "CANDLE_TAKURI",
    "CANDLE_TASUKIGAP",
    "CANDLE_TRISTAR",
    "CANDLE_UNIQUE3RIVER",
    "CANDLE_UPSIDEGAP2CROWS",
    "CANDLE_XSIDEGAP3METHODS",
    "CCI",
    "CMO",
    "CORREL",
    "DEMA",
    "DX",
    "EMA_200",
    "EMA_21",
    "EMA_50",
    "EMA_9",
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_LEADSINE",
    "HT_PHASOR_inphase",
    "HT_PHASOR_quadrature",
    "HT_SINE",
    "HT_TRENDLINE",
    "HT_TRENDMODE",
    "KAMA",
    "LINEARREG",
    "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT",
    "LINEARREG_SLOPE",
    "MACD",
    "MACDEXT",
    "MACDEXT_Hist",
    "MACDEXT_Signal",
    "MACDFIX",
    "MACDFIX_Hist",
    "MACDFIX_Signal",
    "MACD_Hist",
    "MACD_Signal",
    "MAMA",
    "MAX",
    "MAXINDEX",
    "MEDPRICE",
    "MFI",
    "MIDPOINT",
    "MIDPRICE",
    "MIN",
    "MININDEX",
    "MINMAXINDEX_Max",
    "MINMAXINDEX_Min",
    "MINMAX_max",
    "MINMAX_min",
    "MINUS_DI",
    "MINUS_DM",
    "MOM",
    "NATR",
    "OBV",
    "PLUS_DI",
    "PLUS_DM",
    "PPO",
    "ROC",
    "ROCP",
    "ROCR",
    "ROCR100",
    "RSI",
    "SAR",
    "SAREXT",
    "SMA",
    "STDDEV",
    "STOCHF_Fast_D",
    "STOCHF_Fast_K",
    "STOCHRSI_Fast_D",
    "STOCHRSI_Fast_K",
    "STOCH_Slow_D",
    "STOCH_Slow_K",
    "SUM",
    "T3",
    "TEMA",
    "TRANGE",
    "TRIMA",
    "TRIX",
    "TSF",
    "TYPPRICE",
    "ULTOSC",
    "VAR",
    "WCLPRICE",
    "WILLR",
    "WMA",
]
