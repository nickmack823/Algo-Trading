{
  "name": "METROFixed",
  "function": METROFixed,
  "signal_function": signal_metro_fixed,
  "raw_function": METROFixed,
  "description": "Computes Wilder's RSI and StepRSI (fast/slow) via backward clamped recursion with adjustable step sizes. Outputs RSI, StepRSI_fast, and StepRSI_slow aligned to df.index.",
  "parameters": {
    "period_rsi": 14,
    "step_size_fast": 5.0,
    "step_size_slow": 15.0
  },
  "parameter_space": {
    "period_rsi": [7, 10, 14, 21, 28],
    "step_size_fast": [3.0, 5.0, 7.0, 10.0],
    "step_size_slow": [10.0, 15.0, 20.0, 30.0]
  },
  "role": "Momentum"
}