{
  "name": "QQEWithAlerts",
  "function": QQEWithAlerts,
  "signal_function": signal_qqe_with_alerts,
  "raw_function": QQEWithAlerts,
  "description": "QQE smooths RSI with an EMA and adds a dynamic trailing level to form a hybrid oscillator with trend-following characteristics.",
  "parameters": {"rsi_period": 14, "sf": 5},
  "parameter_space": {
    "rsi_period": [7, 14, 21, 30],
    "sf": [3, 5, 7, 9]
  },
  "role": "Confirmation"
}