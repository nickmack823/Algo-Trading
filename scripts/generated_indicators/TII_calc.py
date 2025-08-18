import numpy as np
import pandas as pd

def TII(
    df: pd.DataFrame,
    length: int = 30,
    ma_length: int = 60,
    ma_method: int = 0,  # 0=SMA, 1=EMA, 2=SMMA, 3=LWMA
    price: int = 0       # 0=Close,1=Open,2=High,3=Low,4=Median,5=Typical,6=Weighted
) -> pd.Series:
    """Trend Intensity Index (TII) as in MQL4 TII.mq4.
    Returns a Series aligned to df.index with NaNs for warmup.
    """
    if length <= 0 or ma_length <= 0:
        return pd.Series(np.nan, index=df.index, name="TII")

    # Applied price
    if price == 0:
        pr = df["Close"].astype(float)
    elif price == 1:
        pr = df["Open"].astype(float)
    elif price == 2:
        pr = df["High"].astype(float)
    elif price == 3:
        pr = df["Low"].astype(float)
    elif price == 4:
        pr = ((df["High"] + df["Low"]) / 2.0).astype(float)  # Median Price (HL/2)
    elif price == 5:
        pr = ((df["High"] + df["Low"] + df["Close"]) / 3.0).astype(float)  # Typical Price (HLC/3)
    elif price == 6:
        pr = ((df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0).astype(float)  # Weighted Price (HLCC/4)
    else:
        pr = df["Close"].astype(float)

    # Moving average helpers
    def _sma(x: pd.Series, p: int) -> pd.Series:
        return x.rolling(window=p, min_periods=p).mean()

    def _ema(x: pd.Series, p: int) -> pd.Series:
        y = x.ewm(span=p, adjust=False).mean()
        mask = x.notna().rolling(window=p, min_periods=p).count() >= p
        return y.where(mask)

    def _smma(x: pd.Series, p: int) -> pd.Series:
        # Wilder/Smoothed MA with alpha=1/p; mask first p-1 values
        y = x.ewm(alpha=1.0 / p, adjust=False).mean()
        mask = x.notna().rolling(window=p, min_periods=p).count() >= p
        return y.where(mask)

    def _lwma(x: pd.Series, p: int) -> pd.Series:
        w = np.arange(1, p + 1, dtype=float)
        w_sum = w.sum()
        def func(a: np.ndarray) -> float:
            if np.isnan(a).any():
                return np.nan
            return float(np.dot(a, w) / w_sum)
        return x.rolling(window=p, min_periods=p).apply(func, raw=True)

    def _ma(x: pd.Series, p: int, method: int) -> pd.Series:
        if method == 0:
            return _sma(x, p)
        elif method == 1:
            return _ema(x, p)
        elif method == 2:
            return _smma(x, p)
        elif method == 3:
            return _lwma(x, p)
        else:
            return _sma(x, p)

    ma = _ma(pr, ma_length, ma_method)

    diff = pr - ma
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)

    pos = up.rolling(window=length, min_periods=length).mean()
    neg = down.rolling(window=length, min_periods=length).mean()

    den = pos + neg
    tii = (100.0 * pos / den).where(den != 0)
    tii.name = "TII"
    return tii