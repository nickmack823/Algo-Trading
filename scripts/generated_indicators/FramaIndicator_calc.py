import numpy as np
import pandas as pd

def FramaIndicator(df: pd.DataFrame, period: int = 10, price_type: int = 0) -> pd.Series:
    """Return FRAMA (Fractal Adaptive Moving Average) as a Series aligned to df.index.
    - period: window length (MQL4 PeriodFRAMA), default 10
    - price_type: 0 Close, 1 Open, 2 High, 3 Low, 4 Median (H+L)/2, 5 Typical (H+L+C)/3, 6 Weighted (H+L+2C)/4
    Vectorized rolling calculations; recursive smoothing requires a minimal loop. NaNs preserved for warmup."""
    if period <= 0:
        raise ValueError("period must be a positive integer")

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)

    if price_type == 1:
        price = open_.copy()
    elif price_type == 2:
        price = high.copy()
    elif price_type == 3:
        price = low.copy()
    elif price_type == 4:
        price = (high + low) / 2.0
    elif price_type == 5:
        price = (high + low + close) / 3.0
    elif price_type == 6:
        price = (high + low + 2.0 * close) / 4.0
    else:
        price = close.copy()

    p = int(period)
    two_p = 2 * p

    # Rolling highs/lows for the three segments
    hi1 = high.rolling(window=p, min_periods=p).max()
    lo1 = low.rolling(window=p, min_periods=p).min()

    # Previous block (shift by period to end at t - p)
    hi2 = hi1.shift(p)
    lo2 = lo1.shift(p)

    # Two-period block
    hi3 = high.rolling(window=two_p, min_periods=two_p).max()
    lo3 = low.rolling(window=two_p, min_periods=two_p).min()

    n1 = (hi1 - lo1) / float(p)
    n2 = (hi2 - lo2) / float(p)
    n3 = (hi3 - lo3) / float(two_p)

    # Compute fractal dimension D and adaptive alpha
    eps = np.finfo(float).eps
    log2 = np.log(2.0)

    n1n2 = (n1 + n2).to_numpy(dtype=float)
    denom = n3.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        D = (np.log(np.maximum(n1n2, eps)) - np.log(np.maximum(denom, eps))) / log2
        alpha = np.exp(-4.6 * (D - 1.0))

    pr = price.to_numpy(dtype=float)
    n = pr.shape[0]
    out = np.full(n, np.nan, dtype=float)

    # Find first valid index where alpha and price are finite
    valid_mask = np.isfinite(alpha) & np.isfinite(pr)
    if not valid_mask.any():
        return pd.Series(out, index=df.index, name="FRAMA")

    start_idx = int(np.argmax(valid_mask))  # first True

    # Seed with price at first valid point
    out[start_idx] = pr[start_idx]

    # Recursive smoothing with time-varying alpha
    for t in range(start_idx + 1, n):
        if np.isfinite(alpha[t]) and np.isfinite(pr[t]) and np.isfinite(out[t - 1]):
            a = alpha[t]
            out[t] = a * pr[t] + (1.0 - a) * out[t - 1]
        else:
            # If current inputs invalid, carry forward previous value
            out[t] = out[t - 1]

    return pd.Series(out, index=df.index, name="FRAMA")