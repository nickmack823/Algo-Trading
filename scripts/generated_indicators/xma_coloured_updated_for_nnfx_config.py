{
  "name": "XmaColouredUpdatedForNnfx",
  "function": XmaColouredUpdatedForNnfx,
  "signal_function": signal_xma_coloured_updated_for_nnfx,
  "raw_function": XmaColouredUpdatedForNnfx,
  "description": "Moving-average threshold indicator producing Signal, Fl, Up, and Dn series; uses two MAs with a point threshold on an applied price.",
  "parameters": {
    "period": 12,
    "porog": 3,
    "metod": "ema",
    "metod2": "ema",
    "price": "close",
    "tick_size": 0.0001
  },
  "parameter_space": {
    "period": [8, 12, 20, 34, 50],
    "porog": [1, 2, 3, 5, 8],
    "metod": ["sma", "ema", "smma", "lwma"],
    "metod2": ["sma", "ema", "smma", "lwma"],
    "price": ["close", "open", "high", "low", "median", "typical", "weighted"]
  },
  "role": "Confirmation"
}