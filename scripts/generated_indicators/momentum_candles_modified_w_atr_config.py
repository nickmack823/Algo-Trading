indicator_config = {
  "name": "Momentum Candles Modified w ATR",
  "function": MomentumCandlesModifiedWAtr,
  "signal_function": signal_momentum_candles_modified_w_atr,
  "raw_function": MomentumCandlesModifiedWAtr,
  "description": "ATR-normalized candle momentum with static thresholds at +/- (1/atr_multiplier). Returns Value, Threshold_Pos, Threshold_Neg.",
  "parameters": {"atr_period": 50, "atr_multiplier": 2.5},
  "parameter_space": {
    "atr_period": [14, 21, 50, 100],
    "atr_multiplier": [1.5, 2.0, 2.5, 3.0, 4.0]
  },
  "role": "Confirmation"
}