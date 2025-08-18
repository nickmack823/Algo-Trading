{
  "name": "cyber-cycle",
  "function": CyberCycle,
  "signal_function": signal_cyber_cycle,
  "raw_function": CyberCycle,
  "description": "* John Ehlers’ Cyber Cycle indicator applies smoothing and differentiation to extract a cyclical component from price. It oscillates around zero.",
  "parameters": {"alpha": 0.07, "price": "hl2"},
  "parameter_space": {
    "alpha": [0.03, 0.05, 0.07, 0.1, 0.15],
    "price": ["close", "hl2", "hlc3", "ohlc4", "wclose"]
  },
  "role": "Confirmation"
}