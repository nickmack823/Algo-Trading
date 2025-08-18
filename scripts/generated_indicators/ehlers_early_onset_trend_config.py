{
  "name": "Ehlers Early Onset Trend",
  "function": EhlersEarlyOnsetTrend,
  "signal_function": signal_ehlers_early_onset_trend,
  "raw_function": EhlersEarlyOnsetTrend,
  "description": "A John Ehlers indicator designed to detect the early onset of trends using high-pass filtering, adaptive smoothing, and a quotient transform. Returns two lines (EEOT_Q1, EEOT_Q2).",
  "parameters": {"period": 20, "q1": 0.8, "q2": 0.4},
  "parameter_space": {"period": [10, 20, 30, 50], "q1": [0.6, 0.8, 0.9], "q2": [0.2, 0.4, 0.6]},
  "role": "Trend"
}