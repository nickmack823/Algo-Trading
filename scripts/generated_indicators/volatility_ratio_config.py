{
  "name": "Volatility Ratio",
  "function": VolatilityRatio,
  "signal_function": signal_volatility_ratio,
  "raw_function": VolatilityRatio,
  "description": "Volatility Ratio (VR) = rolling std of price divided by its SMA over the same period. Outputs VR plus two alternating segments where VR < 1 (VR_below1_a and VR_below1_b).",
  "parameters": {"period": 25, "price": "Close"},
  "parameter_space": {
    "period": [10, 14, 20, 25, 30, 50],
    "price": ["Close", "HL2", "HLC3", "OHLC4", "Weighted"]
  },
  "role": "ATR/Volatility"
}