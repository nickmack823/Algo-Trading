{
  "name": "precision-trend-histogram",
  "function": PrecisionTrendHistogram,
  "signal_function": signal_precision_trend_histogram,
  "raw_function": PrecisionTrendHistogram,
  "description": "A histogram that visualises trend strength and direction using an average range band and sequential trend state.",
  "parameters": {"avg_period": 30, "sensitivity": 3.0},
  "parameter_space": {
    "avg_period": [10, 20, 30, 50],
    "sensitivity": [1.0, 2.0, 3.0, 4.0]
  },
  "role": "Confirmation"
}