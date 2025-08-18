{
  "name": "SherifHilo",
  "function": SherifHilo,
  "signal_function": signal_sherif_hilo,
  "raw_function": SherifHilo,
  "description": "Rolling Hi-Lo regime indicator that switches Data to LLV or HHV based on close vs previous value; outputs LineUp/LineDown.",
  "parameters": {"period_high": 100, "period_lows": 100},
  "parameter_space": {
    "period_high": [20, 50, 100, 150, 200],
    "period_lows": [20, 50, 100, 150, 200]
  },
  "role": "Confirmation"
}