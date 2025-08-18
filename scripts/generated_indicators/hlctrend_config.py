{
  "name": "HLCTrend",
  "function": Hlctrend,
  "signal_function": signal_hlctrend,
  "raw_function": Hlctrend,
  "description": "Trend indicator using EMAs of Close, Low, and High; outputs two lines: EMA(Close)-EMA(High) and EMA(Low)-EMA(Close).",
  "parameters": {"close_period": 5, "low_period": 13, "high_period": 34},
  "parameter_space": {
    "close_period": [3, 5, 8, 13],
    "low_period": [8, 13, 21, 34],
    "high_period": [21, 34, 55, 89]
  },
  "role": "Trend"
}