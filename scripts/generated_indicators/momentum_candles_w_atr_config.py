{
  "name": "Momentum Candles w ATR",
  "function": MomentumCandlesWATR,
  "signal_function": "signal_momentum_candles_w_atr",
  "raw_function": MomentumCandlesWATR,
  "description": "Colours candles based on momentum with an ATR filter: bullish if Close > Open and ATR(atr_period)/abs(Close-Open) < atr_multiplier; bearish if Close < Open under the same filter.",
  "parameters": {"atr_period": 50, "atr_multiplier": 2.5},
  "parameter_space": {
    "atr_period": [14, 21, 50, 100],
    "atr_multiplier": [1.5, 2.0, 2.5, 3.0, 4.0]
  },
  "role": "Confirmation"
}