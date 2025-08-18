{
  "name": "TrendAkkam",
  "function": TRENDAKKAM,
  "signal_function": signal_trend_akkam,
  "raw_function": TRENDAKKAM,
  "description": "ATR-based trailing stop that flips with price and can use either ATR*factor or a fixed delta for stop distance.",
  "parameters": {
    "akk_range": 100,
    "ima_range": 1,
    "akk_factor": 6.0,
    "mode": 0,
    "delta_price": 30.0,
    "point": 1.0
  },
  "parameter_space": {
    "akk_range": [14, 50, 100, 200],
    "ima_range": [1, 3, 5, 10],
    "akk_factor": [2.0, 3.0, 4.0, 6.0],
    "mode": [0, 1],
    "delta_price": [10.0, 20.0, 30.0, 50.0],
    "point": [0.01, 0.1, 1.0]
  },
  "role": "Trend"
}