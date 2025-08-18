{
  "name": "Silence",
  "function": Silence,
  "signal_function": signal_silence,
  "raw_function": Silence,
  "description": "Silence indicator producing normalized (0..100, reversed) Aggressiveness and Volatility series over rolling windows; higher values indicate quieter conditions.",
  "parameters": {"my_period": 12, "buff_size": 96, "point": 0.0001, "redraw": True},
  "parameter_space": {
    "my_period": [8, 12, 16, 24],
    "buff_size": [64, 96, 128, 192]
  },
  "role": "ATR/Volatility"
}