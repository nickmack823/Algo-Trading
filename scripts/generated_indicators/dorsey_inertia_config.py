{
  "name": "dorsey-inertia",
  "function": DorseyInertia,
  "signal_function": signal_dorsey_inertia,
  "raw_function": DorseyInertia,
  "description": "Dorsey Inertia (Mladen): EMA-averaged up/down rolling-std RVI components with final SMA smoothing.",
  "parameters": {"rvi_period": 10, "avg_period": 14, "smoothing_period": 20},
  "parameter_space": {
    "rvi_period": [5, 8, 10, 14, 20, 30],
    "avg_period": [8, 10, 14, 21, 30],
    "smoothing_period": [10, 14, 20, 30, 50]
  },
  "role": "Trend"
}