{
  "name": "BamsBung3",
  "function": BamsBung3,
  "signal_function": None,
  "raw_function": BamsBung3,
  "description": "SMA-based Bollinger Band stop-and-signal indicator producing up/down trend stops, entry signals, and optional lines.",
  "parameters": {"length": 14, "deviation": 2.0, "money_risk": 0.02, "signal_mode": 1, "line_mode": 1},
  "parameter_space": {
    "length": [10, 14, 20, 30],
    "deviation": [1.5, 2.0, 2.5, 3.0],
    "money_risk": [0.01, 0.02, 0.05, 0.1],
    "signal_mode": [0, 1, 2],
    "line_mode": [0, 1]
  },
  "role": "ATR/Volatility"
}