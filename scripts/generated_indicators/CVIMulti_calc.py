import numpy as np
import pandas as pd

def CVIMulti(df: pd.DataFrame, length: int = 14, method: int = 0, use_modified: bool = False) -> pd.Series:
    """Return the Chartmill Value Indicator (CVI) as a Series aligned to df.index.
    
    Params:
    - length: int = 14 (lookback period)
    - method: int = 0 (0=SMA, 1=EMA, 2=SMMA/RMA, 3=LWMA)
    - use_modified: bool = False (if True, divide by ATR * sqrt(length), else by ATR)
    """
    if length <= 0:
        return pd.Series(np.nan, index=df.index, name="CVI")
    if method not in (0, 1, 2, 3):
        raise ValueError("method must be one of {0: SMA, 1: EMA, 2: SMMA, 3: LWMA}")

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    median_price = (high + low) / 2.0

    def sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n, min_periods=n).mean()

    def ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def smma(s: pd.Series, n: int) -> pd.Series:
        # Wilder's smoothing (RMA)
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    def lwma(s: pd.Series, n: int) -> pd.Series:
        # Linear Weighted MA with weights increasing to the most recent
        arr = s.to_numpy(dtype=float)
        w = np.arange(1, n + 1, dtype=float)
        m = (~np.isnan(arr)).astype(float)
        x = np.where(np.isnan(arr), 0.0, arr)

        num = np.convolve(x, w, mode="full")[n - 1:n - 1 + arr.size]
        den = np.convolve(m, w, mode="full")[n - 1:n - 1 + arr.size]

        full_weight = w.sum()
        out = np.where(den == full_weight, num / den, np.nan)
        return pd.Series(out, index=s.index)

    if method == 0:
        vc = sma(median_price, length)
    elif method == 1:
        vc = ema(median_price, length)
    elif method == 2:
        vc = smma(median_price, length)
    else:
        vc = lwma(median_price, length)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    denom = atr * (np.sqrt(length) if use_modified else 1.0)
    denom = denom.where(denom != 0.0, np.nan)

    cvi = (close - vc) / denom
    cvi.name = "CVI"
    return cvi