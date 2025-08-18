{
  "name": "Price Momentum Oscillator",
  "function": PriceMomentumOscillator,
  "signal_function": signal_price_momentum_oscillator,
  "raw_function": PriceMomentumOscillator,
  "description": "A momentum oscillator that applies two-stage EMA smoothing to percentage price change (alpha=2/one then 2/two), scaled by 10; the signal is an EMA of the PMO. Oscillates around zero.",
  "parameters": {"one": 35, "two": 20, "period": 10},
  "parameter_space": {
    "one": [20, 25, 30, 35, 40, 45],
    "two": [10, 14, 20, 26, 30],
    "period": [5, 10, 14, 20]
  },
  "role": "Momentum"
}