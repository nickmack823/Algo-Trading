{
  "name": "JpOscillator",
  "function": "JpOscillator",
  "signal_function": "signal_jp_oscillator",
  "raw_function": "JpOscillator",
  "description": "Forward-looking buffer 2*Close - 0.5*Close.shift(-1) - 0.5*Close.shift(-2) - Close.shift(-4), optionally smoothed with SMA/EMA/SMMA/LWMA; outputs Jp plus slope-segmented JpUp/JpDown.",
  "parameters": {"period": 5, "mode": 0, "smoothing": True},
  "parameter_space": {"period": [3, 5, 8, 13, 21], "mode": [0, 1, 2, 3], "smoothing": [True, False]},
  "role": "Momentum"
}