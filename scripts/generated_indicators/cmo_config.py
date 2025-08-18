{
  "name": "CMO",
  "function": CMO,
  "signal_function": signal_cmo,
  "raw_function": CMO,
  "description": "Tushar Chande's oscillator measures momentum by comparing the sum of up moves to the sum of down moves over a look-back period.",
  "parameters": {"length": 9, "price": "Close"},
  "parameter_space": {
    "length": [5, 9, 14, 20, 30],
    "price": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"]
  },
  "role": "Momentum"
}