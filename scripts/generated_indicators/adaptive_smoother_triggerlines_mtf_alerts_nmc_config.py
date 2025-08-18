{
  "name": "AdaptiveSmootherTriggerlinesMtfAlertsNmc",
  "function": AdaptiveSmootherTriggerlinesMtfAlertsNmc,
  "signal_function": None,
  "raw_function": AdaptiveSmootherTriggerlinesMtfAlertsNmc,
  "description": "Adaptive smoother (iSmooth) with dynamic period from std-dev ratio; outputs current lsma, previous-bar lsma (lwma), slope-based trends (+1/-1), and optional multi-color segmented down-trend series.",
  "parameters": {"LsmaPeriod": 50, "LsmaPrice": 0, "AdaptPeriod": 21, "MultiColor": True},
  "parameter_space": {
    "LsmaPeriod": [10, 21, 34, 50, 100],
    "LsmaPrice": [0, 1, 2, 3, 4, 5, 6],
    "AdaptPeriod": [10, 14, 21, 30, 50],
    "MultiColor": [True, False]
  },
  "role": "Confirmation"
}