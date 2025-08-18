{
  "name": "Coral",
  "function": Coral,
  "signal_function": signal_coral,
  "raw_function": Coral,
  "description": "* A linear lag removal (LLR) filter that produces a very smooth moving average with minimal delay. Often called the “Coral Trend Indicator.”",
  "parameters": {"length": 34, "coef": 0.4},
  "parameter_space": {"length": [21, 34, 55, 89], "coef": [0.3, 0.4, 0.5]},
  "role": "Trend"
}