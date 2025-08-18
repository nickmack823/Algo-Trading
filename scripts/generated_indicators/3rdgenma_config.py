{
  "name": "3rdgenma",
  "function": "ThirdGenMA",
  "signal_function": "signal_ThirdGenMA",
  "raw_function": "ThirdGenMA",
  "description": "Third Generation Moving Average (Durschner). Two-pass MA with adaptive alpha to reduce lag while preserving smoothness. Supports SMA, EMA, SMMA, LWMA and multiple applied prices; outputs MA3G and MA1.",
  "parameters": {"ma_period": 220, "sampling_period": 50, "method": 1, "applied_price": 5},
  "parameter_space": {
    "ma_period": [180, 220, 260, 300, 360],
    "sampling_period": [30, 40, 50, 60, 70],
    "method": [0, 1, 2, 3],
    "applied_price": [0, 1, 2, 3, 4, 5, 6]
  },
  "role": "Trend"
}