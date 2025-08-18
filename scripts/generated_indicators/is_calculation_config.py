{
  "name": "IS Calculation",
  "function": ISCalculation,
  "signal_function": signal_is_calculation,
  "raw_function": ISCalculation,
  "description": "* A momentum indicator that applies additional smoothing to a traditional momentum calculation (price – price n periods ago).",
  "parameters": {"period": 10, "nbchandelier": 10, "lag": 0},
  "parameter_space": {
    "period": [5, 10, 14, 20, 30],
    "nbchandelier": [5, 10, 14, 20],
    "lag": [0, 1, 2, 3, 5]
  },
  "role": "Momentum"
}