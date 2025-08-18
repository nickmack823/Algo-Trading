{
  "name": "TrendDirectionForceIndexSmoothed4",
  "function": TrendDirectionForceIndexSmoothed4,
  "signal_function": signal_trend_direction_force_index_smoothed_4,
  "raw_function": TrendDirectionForceIndexSmoothed4,
  "description": "Trend Direction Force Index (Smoothed v4): compares a chosen moving average to its EMA-like smooth, normalizes and iSmooth-filters the result, and emits up/down triggers and a trend state.",
  "parameters": {
    "trend_period": 20,
    "trend_method": "ema",
    "price": "close",
    "trigger_up": 0.05,
    "trigger_down": -0.05,
    "smooth_length": 5.0,
    "smooth_phase": 0.0,
    "color_change_on_zero_cross": False,
    "point": 1.0
  },
  "parameter_space": {
    "trend_period": [10, 20, 30, 50],
    "trend_method": ["sma", "ema", "dsema", "dema", "tema", "smma", "lwma", "pwma", "vwma", "hull", "tma", "sine", "mcg", "zlma", "lead", "ssm", "smoo", "linr", "ilinr", "ie2", "nlma"],
    "price": ["close", "open", "high", "low", "median", "typical", "weighted", "average", "medianb", "tbiased", "haclose", "haopen", "hahigh", "halow", "hamedian", "hatypical", "haweighted", "haaverage", "hamedianb", "hatbiased"],
    "trigger_up": [0.02, 0.05, 0.1],
    "trigger_down": [-0.02, -0.05, -0.1],
    "smooth_length": [3.0, 5.0, 8.0, 13.0],
    "smooth_phase": [-50.0, 0.0, 50.0],
    "color_change_on_zero_cross": [False, True],
    "point": [0.01, 0.1, 1.0]
  },
  "role": "Trend"
}