{
  "name": "Band Pass Filter",
  "function": BandPassFilter,
  "signal_function": signal_band_pass_filter,
  "raw_function": BandPassFilter,
  "description": "* Passes price components within a target frequency band (cycle emphasis) while attenuating trend and high-frequency noise; includes slope/trend-segmented histogram outputs.",
  "parameters": {"period": 50, "price": "median", "delta": 0.1},
  "parameter_space": {
    "period": [20, 30, 40, 50, 60, 80],
    "price": ["median", "close", "typical", "weighted", "average", "ha_median", "ha_typical", "ha_weighted", "hab_median"],
    "delta": [0.05, 0.1, 0.15, 0.2]
  },
  "role": "Confirmation"
}