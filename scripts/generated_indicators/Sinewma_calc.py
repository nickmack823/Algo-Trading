import numpy as np
import pandas as pd

def Sinewma(df: pd.DataFrame, length: int = 20, price: int = 0) -> pd.Series:
    """Return the Sine Weighted Moving Average (SineWMA) aligned to df.index.
    - length: window size (default 20, must be >=1)
    - price: applied price mapping:
        0 Close, 1 Open, 2 High, 3 Low, 4 Median(HL/2), 5 Typical(HLC/3), 6 Weighted(HLCC/4)
    Vectorized with numpy; preserves length with NaNs for warmup or incomplete windows; handles NaNs."""
    if length < 1:
        raise ValueError("length must be >= 1")

    # Applied price selection
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    if price == 0:
        x = c
    elif price == 1:
        x = o
    elif price == 2:
        x = h
    elif price == 3:
        x = l
    elif price == 4:
        x = (h + l) / 2.0
    elif price == 5:
        x = (h + l + c) / 3.0
    elif price == 6:
        x = (h + l + 2.0 * c) / 4.0
    else:
        x = c  # fallback to Close

    x_arr = x.to_numpy(dtype=float)
    n = x_arr.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(out, index=df.index, name=f"SineWMA_{length}")

    # Sine weights
    w = np.sin(np.pi * (np.arange(length) + 1.0) / (length + 1.0))
    w_sum = w.sum()
    w_rev = w[::-1]

    # Handle NaNs: require full window of valid values
    valid = np.isfinite(x_arr).astype(float)
    x_filled = np.where(np.isfinite(x_arr), x_arr, 0.0)

    num_full = np.convolve(x_filled, w_rev, mode="full")
    cnt_full = np.convolve(valid, np.ones(length, dtype=float), mode="full")

    end_idx_start = length - 1
    end_idx_stop = n  # inclusive of last index n-1
    vals = num_full[end_idx_start:end_idx_stop] / w_sum
    cnts = cnt_full[end_idx_start:end_idx_stop]

    out[end_idx_start:end_idx_stop] = np.where(cnts == length, vals, np.nan)

    return pd.Series(out, index=df.index, name=f"SineWMA_{length}")