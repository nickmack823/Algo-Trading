{
  "name": "SmoothStep",
  "function": Smoothstep,
  "raw_function": Smoothstep,
  "description": "A smoothing filter that applies the mathematical smoothstep function to price data, creating a very smooth baseline.",
  "parameters": {"period": 32, "price": "close"},
  "parameter_space": {
    "period": [8, 16, 32, 64],
    "price": ["close", "open", "high", "low", "median", "typical", "weighted", "lowhigh"]
  },
  "role": "Baseline"
}