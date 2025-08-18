{
  "name": "ASO",
  "function": ASO,
  "signal_function": signal_aso,
  "raw_function": ASO,
  "description": "* This oscillator attempts to quantify market sentiment by averaging momentum readings from several timeframes.",
  "parameters": {"period": 10, "mode": 0, "bulls": True, "bears": True},
  "parameter_space": {
    "period": [5, 10, 14, 20, 30],
    "mode": [0, 1, 2],
    "bulls": [True, False],
    "bears": [True, False]
  },
  "role": "Confirmation"
}