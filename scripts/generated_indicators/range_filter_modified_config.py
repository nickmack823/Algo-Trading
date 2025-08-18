{
  "name": "RangeFilterModified",
  "function": RangeFilterModified,
  "signal_function": signal_range_filter_modified,
  "raw_function": RangeFilterModified,
  "description": "Dynamic filter using Wilder ATR(atr_period) * multiplier to form adaptive bands around a recursive center line.",
  "parameters": {"atr_period": 14, "multiplier": 3.0},
  "parameter_space": {
    "atr_period": [7, 10, 14, 21, 28],
    "multiplier": [1.5, 2.0, 2.5, 3.0, 3.5]
  },
  "role": "Baseline"
}