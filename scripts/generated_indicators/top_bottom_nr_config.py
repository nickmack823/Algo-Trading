{
  "name": "Top Bottom NR",
  "function": TopBottomNR,
  "signal_function": None,
  "raw_function": TopBottomNR,
  "description": "Computes run-length counts since the most recent breakout of lows/highs over the previous period, returning LongSignal and ShortSignal aligned to the input index.",
  "parameters": {"per": 14},
  "parameter_space": {"per": [7, 14, 21, 30]},
  "role": "Trend"
}