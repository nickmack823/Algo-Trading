{
  "name": "Decycler Oscillator",
  "function": DecyclerOscillator,
  "signal_function": signal_decycler_oscillator,
  "raw_function": DecyclerOscillator,
  "description": "John Ehlers-style decycler oscillator (two-pass) that removes dominant cycle components and highlights trend; includes segmented down-trend buffers.",
  "parameters": {
    "hp_period": 125,
    "k": 1.0,
    "hp_period2": 100,
    "k2": 1.2,
    "price": "close"
  },
  "parameter_space": {
    "hp_period": [75, 100, 125, 150, 200],
    "k": [0.5, 1.0, 1.5, 2.0],
    "hp_period2": [40, 60, 80, 100, 125],
    "k2": [0.8, 1.0, 1.2, 1.5],
    "price": ["close", "median", "typical", "weighted", "average", "tbiased", "haclose", "hamedian", "haweighted"]
  },
  "role": "Confirmation"
}