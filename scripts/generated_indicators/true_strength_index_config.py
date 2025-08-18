{
  "name": "true-strength-index",
  "function": TrueStrengthIndex,
  "signal_function": signal_true_strength_index,
  "raw_function": TrueStrengthIndex,
  "description": "A momentum oscillator created by William Blau that uses double-smoothed EMAs of price changes. TSI oscillates around zero and ranges roughly between -100 and +100.",
  "parameters": {"first_r": 5, "second_s": 8},
  "parameter_space": {
    "first_r": [3, 5, 8, 13, 25],
    "second_s": [5, 8, 13, 21]
  },
  "role": "Momentum"
}