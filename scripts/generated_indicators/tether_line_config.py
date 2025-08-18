{
  "name": "TetherLine",
  "function": TetherLine,
  "signal_function": None,
  "raw_function": TetherLine,
  "description": "Baseline from the midpoint of rolling Highest High and Lowest Low over length. Outputs AboveCenter when Close > midpoint and BelowCenter when Close < midpoint; ArrowUp/ArrowDown are NaN placeholders.",
  "parameters": {"length": 55},
  "parameter_space": {"length": [21, 34, 55, 89]},
  "role": "Baseline"
}