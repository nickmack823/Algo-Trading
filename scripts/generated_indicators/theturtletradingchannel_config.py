{
  "name": "TheTurtleTradingChannel",
  "function": TheTurtleTradingChannel,
  "signal_function": signal_TheTurtleTradingChannel,
  "raw_function": TheTurtleTradingChannel,
  "description": "* Based on the Turtle Trading rules, this channel plots recent highs and lows (Donchian channels) to define breakout levels.",
  "parameters": {"trade_period": 20, "stop_period": 10, "strict": False},
  "parameter_space": {
    "trade_period": [10, 20, 40, 55],
    "stop_period": [5, 10, 15, 20],
    "strict": [False, True]
  },
  "role": "Trend"
}