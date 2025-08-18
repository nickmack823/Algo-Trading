{
  "name": "TII",
  "function": TII,
  "signal_function": signal_tii,
  "raw_function": TII,
  "description": "Trend Intensity Index (TII). Measures the strength of a trend relative to a moving average over a look-back window, ranging from 0 to 100.",
  "parameters": {"length": 30, "ma_length": 60, "ma_method": 0, "price": 0},
  "parameter_space": {
    "length": [14, 20, 30, 40, 50],
    "ma_length": [30, 60, 90, 120],
    "ma_method": [0, 1, 2, 3],
    "price": [0, 1, 2, 3, 4, 5, 6]
  },
  "role": "Confirmation"
}