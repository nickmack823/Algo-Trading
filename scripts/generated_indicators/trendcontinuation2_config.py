{
  "name": "TrendContinuation2",
  "function": Trendcontinuation2,
  "signal_function": None,
  "raw_function": Trendcontinuation2,
  "description": "* An indicator designed to confirm whether an existing trend is continuing or faltering by blending moving averages and momentum.",
  "parameters": {"n": 20, "t3_period": 5, "b": 0.618},
  "parameter_space": {
    "n": [10, 20, 30, 50],
    "t3_period": [3, 5, 8, 10],
    "b": [0.5, 0.618, 0.7, 0.8]
  },
  "role": "Confirmation"
}