{
  "name": "GlitchIndexFixed",
  "function": GlitchIndexFixed,
  "signal_function": signal_glitch_index_fixed,
  "raw_function": GlitchIndexFixed,
  "description": "* An obscure momentum indicator designed to detect unusual “glitches” or anomalies in price movement.  It acts as a smoothed oscillator.",
  "parameters": {
    "MaPeriod": 30,
    "MaMethod": "sma",
    "Price": "median",
    "level1": 1.0,
    "level2": 1.0
  },
  "parameter_space": {
    "MaPeriod": [10, 20, 30, 50, 100],
    "MaMethod": ["sma", "ema", "smma", "lwma", "slwma", "dsema", "tema", "lsma", "nlma"],
    "Price": ["close", "median", "typical", "weighted", "haaverage"],
    "level1": [0.5, 1.0, 1.5],
    "level2": [1.5, 2.0, 2.5]
  },
  "role": "Momentum"
}