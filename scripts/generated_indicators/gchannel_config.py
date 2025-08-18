{
  "name": "GChannel",
  "function": Gchannel,
  "raw_function": Gchannel,
  "description": "GChannel plots a recursive dynamic price channel (Upper/Middle/Lower). The midline can serve as a baseline; the channel adapts to volatility.",
  "parameters": {"length": 100, "price": "close"},
  "parameter_space": {
    "length": [20, 50, 100, 150, 200],
    "price": ["close", "open", "high", "low", "median", "typical", "weighted"]
  },
  "role": "Baseline"
}