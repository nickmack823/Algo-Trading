{
  "name": "TTF",
  "function": TTF,
  "signal_function": None,
  "raw_function": TTF,
  "description": "Trend Trigger Factor (TTF) from rolling highs/lows with T3 smoothing; returns TTF and a threshold Signal at top_line/bottom_line.",
  "parameters": {"ttf_bars": 8, "top_line": 75.0, "bottom_line": -75.0, "t3_period": 3, "b": 0.7},
  "parameter_space": {
    "ttf_bars": [5, 8, 12, 20],
    "top_line": [60.0, 70.0, 75.0, 80.0],
    "bottom_line": [-80.0, -75.0, -70.0, -60.0],
    "t3_period": [1, 2, 3, 5],
    "b": [0.5, 0.6, 0.7, 0.8]
  },
  "role": "Confirmation"
}