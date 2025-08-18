{
  "name": "ErgodicTVI",
  "function": ErgodicTVI,
  "signal_function": None,
  "raw_function": ErgodicTVI,
  "description": "* This oscillator combines price momentum and volatility information to produce an “ergodic” signal line with a smoothing component. It is similar to an MACD of the True Volume Indicator.",
  "parameters": {
    "Period1": 12,
    "Period2": 12,
    "Period3": 1,
    "EPeriod1": 5,
    "EPeriod2": 5,
    "EPeriod3": 5,
    "pip_size": 0.0001
  },
  "parameter_space": {
    "Period1": [8, 12, 16, 20, 34],
    "Period2": [8, 12, 16, 20, 34],
    "Period3": [1, 2, 3, 5],
    "EPeriod1": [3, 5, 8, 13],
    "EPeriod2": [3, 5, 8, 13],
    "EPeriod3": [3, 5, 8, 13]
  },
  "role": "VolumeIndicator"
}