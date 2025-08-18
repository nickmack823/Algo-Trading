{
  "name": "trend-lord-nrp-indicator",
  "function": TrendLordNrpIndicator,
  "signal_function": signal_trend_lord_nrp_indicator,
  "raw_function": TrendLordNrpIndicator,
  "description": "Two-stage moving average (length and sqrt(length)) on selected price. Buy slots to Low/High (or MA) based on up/down; Sell is the second-stage MA.",
  "parameters": {"length": 12, "mode": "smma", "price": "close", "show_high_low": False},
  "parameter_space": {
    "length": [8, 12, 16, 20, 24, 30],
    "mode": ["sma", "ema", "smma", "lwma"],
    "price": ["close", "open", "high", "low", "median", "typical", "weighted"],
    "show_high_low": [False, True]
  },
  "role": "Trend"
}