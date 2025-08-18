{
  "name": "Ehlers DELI (Detrended Leading Indicator)",
  "function": EhlersDELIDetrendedLeadingIndicator,
  "signal_function": signal_ehlers_deli_detrended_leading_indicator,
  "raw_function": EhlersDELIDetrendedLeadingIndicator,
  "description": "* A detrended oscillator by John Ehlers that leads price by removing lag and emphasising turning points.",
  "parameters": {"period": 14},
  "parameter_space": {"period": [7, 14, 21, 30]},
  "role": "Confirmation"
}