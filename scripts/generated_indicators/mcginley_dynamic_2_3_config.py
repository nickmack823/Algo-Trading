{
  "name": "McGinley Dynamic 2.3",
  "function": McginleyDynamic23,
  "signal_function": signal_mcginley_dynamic_2_3,
  "raw_function": McginleyDynamic23,
  "description": "Created by J. R. McGinley, this indicator adjusts its smoothing factor dynamically based on the speed of price changes. Unlike a fixed-period moving average, the McGinley Dynamic speeds up in fast markets and slows down in slow markets.",
  "parameters": {"period": 12, "price": "close", "constant": 5.0, "method": "ema"},
  "parameter_space": {
    "period": [5, 8, 12, 14, 20, 24, 30],
    "price": ["close", "open", "high", "low", "median", "typical", "weighted"],
    "constant": [3.0, 5.0, 8.0, 10.0],
    "method": ["sma", "ema", "smma", "lwma", "gen"]
  },
  "role": "Baseline"
}