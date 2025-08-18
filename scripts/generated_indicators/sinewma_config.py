{
  "name": "SineWMA",
  "function": Sinewma,
  "signal_function": signal_sinewma,
  "raw_function": Sinewma,
  "description": "A moving average that applies half-sine weights across the lookback window. The highest weights are centered, providing a smooth curve that remains responsive to recent price while filtering noise.",
  "parameters": {"length": 20, "price": 0},
  "parameter_space": {"length": [10, 14, 20, 30, 50], "price": [0, 1, 2, 3, 4, 5, 6]},
  "role": "Baseline"
}