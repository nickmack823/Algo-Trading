{
  "name": "Ehlers Reverse EMA",
  "function": EhlersReverseEMA,
  "signal_function": signal_ehlers_reverse_ema,
  "raw_function": EhlersReverseEMA,
  "description": "* A variant of the exponential moving average that applies weighting in reverse order (most recent prices receive the least weight). This aims to anticipate turning points.",
  "parameters": {"alpha": 0.1},
  "parameter_space": {"alpha": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]},
  "role": "Confirmation"
}