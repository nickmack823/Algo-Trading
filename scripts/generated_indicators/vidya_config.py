{
  "name": "VIDYA",
  "function": VIDYA,
  "signal_function": signal_vidya,
  "raw_function": VIDYA,
  "description": "VIDYA is an adaptive moving average that scales an EMA by a volatility ratio (short std divided by long std). Uses Close price only.",
  "parameters": {"period": 9, "histper": 30},
  "parameter_space": {"period": [5, 9, 14, 21], "histper": [20, 30, 50, 100]},
  "role": "Trend"
}