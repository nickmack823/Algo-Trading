{
  "name": "TP",
  "function": TP,
  "signal_function": None,
  "raw_function": TP,
  "description": "Advance Trend Pressure (TP). Computes rolling sums of up/down contributions over 'length' bars; TP = Up - Dn. Optionally outputs Up and Dn lines when show_updn=True.",
  "parameters": {"length": 14, "show_updn": False},
  "parameter_space": {"length": [7, 14, 21, 30], "show_updn": [False, True]},
  "role": "Trend"
}