{
  "name": "ChandelierExit",
  "function": Chandelierexit,
  "signal_function": signal_chandelierexit,
  "raw_function": Chandelierexit,
  "description": "* A trailing stop indicator developed by Chuck LeBeau.  It places the stop a multiple of the Average True Range (ATR) away from the highest high (for long trades) or lowest low (for short trades).",
  "parameters": {
    "lookback": 7,
    "atr_period": 9,
    "atr_mult": 2.5,
    "shift": 0
  },
  "parameter_space": {
    "lookback": [5, 7, 10, 14, 20, 30],
    "atr_period": [7, 9, 14, 21],
    "atr_mult": [2.0, 2.5, 3.0, 3.5]
  },
  "role": "Exit"
}