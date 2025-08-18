{
  "name": "Trimagen",
  "function": Trimagen,
  "raw_function": Trimagen,
  "description": "Triangular Moving Average (TriMA): an SMA of an SMA, center-weighted smoothing that is smoother than a standard SMA.",
  "parameters": {"period": 20, "applied_price": "close"},
  "parameter_space": {
    "period": [10, 14, 20, 30, 50],
    "applied_price": ["close", "open", "high", "low", "median", "typical", "weighted"]
  },
  "role": "Baseline"
}