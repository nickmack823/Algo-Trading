{
  "name": "Stiffness Indicator",
  "function": StiffnessIndicator,
  "raw_function": StiffnessIndicator,
  "description": "Measures price stiffness by comparing Close to a thresholded MA and summing signals; outputs Stiffness and Signal.",
  "parameters": {
    "period1": 100,
    "method1": "sma",
    "period3": 60,
    "period2": 3,
    "method2": "sma"
  },
  "parameter_space": {
    "period1": [50, 100, 150, 200],
    "method1": ["sma", "ema", "smma", "lwma"],
    "period3": [30, 60, 90, 120],
    "period2": [2, 3, 5, 8],
    "method2": ["sma", "ema", "smma", "lwma"]
  },
  "role": "ATR/Volatility"
}