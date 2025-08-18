{
  "name": "SuperTrend",
  "function": Supertrend,
  "signal_function": signal_supertrend,
  "raw_function": Supertrend,
  "description": "* A trend‑following indicator based on ATR and moving averages. It plots a line above or below price and flips direction when price breaks through.",
  "parameters": {"period": 10, "multiplier": 3.0},
  "parameter_space": {"period": [7, 10, 14, 21], "multiplier": [2.0, 2.5, 3.0, 3.5]},
  "role": "Trend"
}