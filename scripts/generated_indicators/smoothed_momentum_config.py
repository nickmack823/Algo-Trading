{
  "name": "SmoothedMomentum",
  "function": SmoothedMomentum,
  "signal_function": signal_smoothed_momentum,
  "raw_function": SmoothedMomentum,
  "description": "Percentage momentum (100 * price / price[n] ago) with optional smoothing via SMA/EMA/SMMA/LWMA on a chosen applied price; returns both smoothed (SM) and raw Momentum.",
  "parameters": {
    "momentum_length": 12,
    "use_smoothing": True,
    "smoothing_method": 0,
    "smoothing_length": 20,
    "price": 0
  },
  "parameter_space": {
    "momentum_length": [5, 10, 12, 14, 20],
    "use_smoothing": [True, False],
    "smoothing_method": [0, 1, 2, 3],
    "smoothing_length": [5, 10, 20, 50, 100],
    "price": [0, 1, 2, 3, 4, 5, 6]
  },
  "role": "Momentum"
}