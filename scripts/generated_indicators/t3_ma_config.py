{
  "name": "T3MA",
  "function": T3MA,
  "signal_function": signal_t3_ma,
  "raw_function": T3MA,
  "description": "The T3 moving average applies a triple smoothing process using exponential moving averages, offering a smoother curve with less lag than simple or double EMAs.",
  "parameters": {"length": 10, "b": 0.88, "price": 0},
  "parameter_space": {
    "length": [5, 8, 10, 14, 20, 30],
    "b": [0.7, 0.8, 0.88, 0.9, 0.95]
  },
  "role": "Baseline"
}