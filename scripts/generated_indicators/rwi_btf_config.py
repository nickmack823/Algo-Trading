{
  "name": "RWI BTF",
  "function": RWIBTF,
  "signal_function": signal_rwi_btf,
  "raw_function": RWIBTF,
  "description": "Random Walk Index (BTF-capable): measures deviation from a random walk by comparing price moves to ATR-scaled expected range.",
  "parameters": {"length": 2, "tf": None},
  "parameter_space": {"length": [2, 3, 4, 6, 8, 10, 14], "tf": [None, "5T", "15T", "1H", "4H", "1D"]},
  "role": "Confirmation"
}