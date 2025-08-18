{
  "name": "Metro-Advanced",
  "function": MetroAdvanced,
  "signal_function": signal_metro_advanced,
  "raw_function": MetroAdvanced,
  "description": "RSI-based step oscillator with fast/slow step channels and adaptive levels; provides a simple metro-style trend/momentum display.",
  "parameters": {
    "period_rsi": 14,
    "rsi_type": "rsi",
    "price": "close",
    "step_size_fast": 5.0,
    "step_size_slow": 15.0,
    "over_sold": 10.0,
    "over_bought": 90.0,
    "minmax_period": 49
  },
  "parameter_space": {
    "period_rsi": [7, 14, 21, 28],
    "rsi_type": ["rsi", "wilder", "rsx", "cutler"],
    "price": ["close", "open", "high", "low", "median", "typical", "weighted", "median_body", "average", "trend_biased", "volume"],
    "step_size_fast": [3.0, 5.0, 7.0],
    "step_size_slow": [10.0, 15.0, 20.0, 25.0],
    "over_sold": [5.0, 10.0, 20.0],
    "over_bought": [80.0, 90.0, 95.0],
    "minmax_period": [21, 34, 49, 63]
  },
  "role": "Trend"
}