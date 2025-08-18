{
  "name": "TTMS",
  "function": TTMS,
  "signal_function": None,
  "raw_function": TTMS,
  "description": "A volatility indicator similar to the TTM Squeeze. It measures the relationship between Bollinger Bands and Keltner Channels to identify low-volatility squeeze conditions.",
  "parameters": {
    "bb_length": 20,
    "bb_deviation": 2.0,
    "keltner_length": 20,
    "keltner_smooth_length": 20,
    "keltner_smooth_method": 0,
    "keltner_deviation": 2.0
  },
  "parameter_space": {
    "bb_length": [10, 14, 20, 30, 50],
    "bb_deviation": [1.0, 1.5, 2.0, 2.5, 3.0],
    "keltner_length": [10, 14, 20, 30, 50],
    "keltner_smooth_length": [10, 14, 20, 30, 50],
    "keltner_smooth_method": [0, 1, 2, 3],
    "keltner_deviation": [1.0, 1.5, 2.0, 2.5]
  },
  "role": "ATR/Volatility"
}