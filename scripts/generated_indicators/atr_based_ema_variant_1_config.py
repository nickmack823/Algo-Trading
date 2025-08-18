{
  "name": "ATRBasedEMAVariant1",
  "function": ATRBasedEMAVariant1,
  "signal_function": lambda df, ema_fastest=14.0, multiplier=300.0: ATRBasedEMAVariant1(df, ema_fastest=ema_fastest, multiplier=multiplier)["EMA_ATR_var1"],
  "raw_function": ATRBasedEMAVariant1,
  "description": "EMA on Close with a dynamic period driven by an EMA(14) of High/Low (ATR proxy). Higher volatility increases the equivalent EMA period, making the baseline slower. Outputs columns ['EMA_ATR_var1', 'EMA_Equivalent'].",
  "parameters": {"ema_fastest": 14.0, "multiplier": 300.0},
  "parameter_space": {
    "ema_fastest": [7.0, 10.0, 14.0, 21.0, 28.0],
    "multiplier": [100.0, 200.0, 300.0, 400.0, 500.0]
  },
  "role": "Baseline"
}