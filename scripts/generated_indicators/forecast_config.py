{
  "name": "forecast",
  "function": Forecast,
  "signal_function": signal_forecast,
  "raw_function": Forecast,
  "description": "* The Forecast Oscillator measures the difference between the actual price and its regression forecast value, expressed as a percentage of price.",
  "parameters": {"length": 20, "price": 0},
  "parameter_space": {"length": [10, 14, 20, 30, 50], "price": [0, 1, 2, 3, 4, 5, 6]},
  "role": "Confirmation"
}