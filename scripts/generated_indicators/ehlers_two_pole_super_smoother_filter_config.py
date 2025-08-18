{
  "name": "EhlersTwoPoleSuperSmootherFilter",
  "function": EhlersTwoPoleSuperSmootherFilter,
  "signal_function": signal_ehlers_two_pole_super_smoother_filter,
  "raw_function": EhlersTwoPoleSuperSmootherFilter,
  "description": "John Ehlers' Super Smoother is a two-pole low-pass filter designed to remove noise with minimal lag. Uses Open price, restarts on NaN segments, and preserves length.",
  "parameters": {"cutoff_period": 15},
  "parameter_space": {"cutoff_period": [10, 15, 20, 30, 40]},
  "role": "Baseline"
}