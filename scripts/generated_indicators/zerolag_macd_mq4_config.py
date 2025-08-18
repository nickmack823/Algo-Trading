{
  "name": "Zerolag_MACD.mq4",
  "function": ZerolagMACDMq4,
  "signal_function": signal_zerolag_macd_mq4,
  "raw_function": ZerolagMACDMq4,
  "description": "Zero-lag MACD using MT4-style EMA seeding; outputs ZL_MACD and ZL_Signal aligned to input index.",
  "parameters": {"fast": 12, "slow": 24, "signal": 9},
  "parameter_space": {"fast": [5, 8, 12, 15], "slow": [20, 24, 26, 30], "signal": [5, 9, 12]},
  "role": "Momentum"
}