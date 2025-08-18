{
  "name": "frama-indicator",
  "function": FramaIndicator,
  "signal_function": signal_frama_indicator,
  "raw_function": FramaIndicator,
  "description": "Developed by John Ehlers, the Fractal Adaptive Moving Average adapts its smoothing via the fractal dimension: faster in trends, slower in choppy markets.",
  "parameters": {"period": 10, "price_type": 0},
  "parameter_space": {"period": [5, 10, 14, 20, 30], "price_type": [0, 1, 2, 3, 4, 5, 6]},
  "role": "Trend"
}