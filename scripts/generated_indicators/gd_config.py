{
  "name": "Gd",
  "function": Gd,
  "signal_function": None,
  "raw_function": Gd,
  "description": "Generalized DEMA-style baseline: GD = (1+vf)*EMA - vf*EMA_of_EMA. Returns ['GD','EMA'] aligned to df.index; supports applied price selection.",
  "parameters": {"length": 20, "vf": 0.7, "price": 0},
  "parameter_space": {
    "length": [10, 20, 30, 50],
    "vf": [0.3, 0.5, 0.7, 1.0],
    "price": [0, 1, 2, 3, 4, 5, 6]
  },
  "role": "Baseline"
}