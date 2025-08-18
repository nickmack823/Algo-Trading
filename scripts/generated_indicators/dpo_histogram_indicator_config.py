config = {
    "name": "dpo-histogram-indicator",
    "function": DpoHistogramIndicator,
    "signal_function": None,
    "raw_function": DpoHistogramIndicator,
    "description": "Detrended Price Oscillator with up/down histograms. Computes price minus a forward-shifted moving average; positive values map to DPO_Up and negatives to DPO_Dn.",
    "parameters": {"period": 14, "ma": "sma"},
    "parameter_space": {
        "period": [10, 14, 20, 30, 50],
        "ma": ["sma", "ema", "smma", "wma"],
    },
    "role": "Momentum",
}