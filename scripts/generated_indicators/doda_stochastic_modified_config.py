{
  "name": "Doda-Stochastic-Modified",
  "function": DodaStochasticModified,
  "signal_function": signal_doda_stochastic_modified,
  "raw_function": DodaStochasticModified,
  "description": "A modified stochastic oscillator: EMA of Close -> stochastic (0-100) over Pds -> EMA (Slw) -> signal EMA (Slwsignal).",
  "parameters": {"Slw": 8, "Pds": 13, "Slwsignal": 9},
  "parameter_space": {
    "Slw": [5, 8, 13, 21],
    "Pds": [9, 13, 21, 34],
    "Slwsignal": [5, 9, 13, 18]
  },
  "role": "Confirmation"
}