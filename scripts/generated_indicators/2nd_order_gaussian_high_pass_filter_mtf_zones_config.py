config = {
    "name": "2nd Order Gaussian High Pass Filter MTF Zones",
    "function": SecondOrderGaussianHighPassFilterMtfZones,
    "signal_function": signal_second_order_gaussian_high_pass_filter_mtf_zones,
    "raw_function": SecondOrderGaussianHighPassFilterMtfZones,
    "description": "* A high-pass filter using Gaussian coefficients to remove low-frequency (trend) components and emphasise short-term fluctuations.",
    "parameters": {
        "alpha": 0.14,
        "timeframe": None,
        "interpolate": True,
        "maxbars": 2000,
    },
    "parameter_space": {
        "alpha": [0.05, 0.1, 0.14, 0.2, 0.3],
        "timeframe": [None, "5T", "15T", "1H", "4H", "1D"],
        "interpolate": [True, False],
    },
    "role": "Confirmation",
}