{
  "name": "William Vix-Fix",
  "function": WilliamVixFix,
  "signal_function": signal_william_vix_fix,
  "raw_function": WilliamVixFix,
  "description": "* A volatility indicator created by Larry Williams that emulates the CBOE VIX using price data. It identifies spikes in fear or complacency.",
  "parameters": {"period": 22},
  "parameter_space": {"period": [7, 14, 20, 22, 30]},
  "role": "ATR/Volatility"
}