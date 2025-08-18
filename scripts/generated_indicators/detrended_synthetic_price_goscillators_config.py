{
  "name": "DetrendedSyntheticPriceGoscillators",
  "function": DetrendedSyntheticPriceGoscillators,
  "signal_function": signal_detrended_synthetic_price_goscillators,
  "raw_function": DetrendedSyntheticPriceGoscillators,
  "description": "* This indicator subtracts a moving average from a “synthetic price” (a weighted price) to remove trend and highlight cycles.",
  "parameters": {
    "dsp_period": 14,
    "price_mode": "median",
    "signal_period": 9,
    "color_on": "outer"
  },
  "parameter_space": {
    "dsp_period": [7, 14, 21, 30],
    "price_mode": [
      "close", "open", "high", "low",
      "median", "medianb", "typical", "weighted", "average",
      "tbiased", "tbiased2",
      "haclose", "haopen", "hahigh", "halow",
      "hamedian", "hamedianb", "hatypical", "haweighted", "haaverage",
      "hatbiased", "hatbiased2"
    ],
    "signal_period": [5, 9, 13, 21],
    "color_on": ["outer", "outer2", "zero", "slope"]
  },
  "role": "Confirmation"
}