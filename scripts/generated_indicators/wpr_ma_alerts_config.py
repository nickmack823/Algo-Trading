{
  "name": "wpr + ma (alerts)",
  "function": WprMaAlerts,
  "signal_function": signal_wpr_ma_alerts,
  "raw_function": WprMaAlerts,
  "description": "Williams %R smoothed by a selectable moving average (sma/ema/smma/lwma), returning WPR, Signal, and Cross states.",
  "parameters": {"wpr_period": 35, "signal_period": 21, "ma_method": "smma"},
  "parameter_space": {
    "wpr_period": [14, 21, 35, 55],
    "signal_period": [5, 9, 13, 21, 34],
    "ma_method": ["sma", "ema", "smma", "lwma"]
  },
  "role": "Confirmation"
}