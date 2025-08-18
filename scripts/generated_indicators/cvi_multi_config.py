{
  "name": "CVI_Multi",
  "function": CVIMulti,
  "signal_function": None,
  "raw_function": CVIMulti,
  "description": "Chartmill Value Indicator (CVI): normalized deviation of Close from a moving average of Median Price ((High+Low)/2). method selects the MA (0=SMA, 1=EMA, 2=SMMA/RMA, 3=LWMA). Denominator is ATR; if use_modified is True, ATR is scaled by sqrt(length).",
  "parameters": {"length": 14, "method": 0, "use_modified": False},
  "parameter_space": {"length": [7, 14, 21, 30], "method": [0, 1, 2, 3], "use_modified": [False, True]},
  "role": "ATR/Volatility"
}