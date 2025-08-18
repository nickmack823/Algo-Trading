import numpy as np
import pandas as pd

def VIDYA(df: pd.DataFrame, period: int = 9, histper: int = 30) -> pd.Series:
    """Return the VIDYA line aligned to df.index; vectorized where possible, stable defaults.
    Parameters:
    - period: int = 9
    - histper: int = 30
    Uses Close price only (as per source)."""
    close = pd.to_numeric(df['Close'], errors='coerce')
    n = len(close)
    if n == 0:
        return pd.Series([], index=df.index, dtype=float, name='VIDYA')

    period = max(1, int(period))
    histper = max(1, int(histper))

    std_short = close.rolling(window=period, min_periods=period).std(ddof=0)
    std_long = close.rolling(window=histper, min_periods=histper).std(ddof=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = std_short / std_long

    sc = 2.0 / (period + 1.0)
    alpha = k * sc  # may exceed 1; matching source behavior

    c = close.to_numpy(dtype=float, copy=False)
    a = alpha.to_numpy(dtype=float, copy=False)
    out = np.full(n, np.nan, dtype=float)

    if histper <= n:
        out[:histper] = c[:histper]
        start = histper
    else:
        out[:] = c
        return pd.Series(out, index=df.index, name='VIDYA')

    for i in range(start, n):
        prev = out[i - 1]
        x = c[i]
        ai = a[i]
        if not np.isfinite(ai):
            ai = 0.0
        if np.isnan(prev):
            out[i] = x
        else:
            out[i] = ai * x + (1.0 - ai) * prev

    return pd.Series(out, index=df.index, name='VIDYA')