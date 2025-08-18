import numpy as np
import pandas as pd

def JpOscillator(df: pd.DataFrame, period: int = 5, mode: int = 0, smoothing: bool = True) -> pd.DataFrame:
    """Return JpOscillator lines as columns ['Jp','JpUp','JpDown'], aligned to df.index.
    
    Params:
    - period: int, moving average length for smoothing (Period1 in MQL4). Default 5.
    - mode: int, MA type over buffer (Mode1 in MQL4): 0=SMA, 1=EMA, 2=SMMA (Wilder/Smoothed), 3=LWMA. Default 0.
    - smoothing: bool, if True apply MA on buffer; if False use raw buffer.
    """
    close = df['Close'].astype(float)

    # Buffer calculation (uses forward-looking shifts like MQL4 indexing)
    c0 = close
    c1 = close.shift(-1)
    c2 = close.shift(-2)
    c4 = close.shift(-4)
    buff = 2.0 * c0 - 0.5 * c1 - 0.5 * c2 - c4

    # Helper MA functions (iMAOnArray equivalents)
    def _sma(s: pd.Series, n: int) -> pd.Series:
        if n <= 1:
            return s.copy()
        return s.rolling(n, min_periods=n).mean()

    def _ema(s: pd.Series, n: int) -> pd.Series:
        if n <= 1:
            return s.copy()
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def _smma(s: pd.Series, n: int) -> pd.Series:
        # MT4 Smoothed MA (Wilder). Seeded with SMA(n), then recursive.
        if n <= 1:
            return s.copy()
        arr = s.to_numpy(dtype=float)
        out = np.full(arr.shape[0], np.nan, dtype=float)
        sma = pd.Series(arr).rolling(n, min_periods=n).mean().to_numpy()
        idxs = np.where(np.isfinite(sma))[0]
        if idxs.size == 0:
            return pd.Series(out, index=s.index)
        start = idxs[0]
        out[start] = sma[start]
        for i in range(start + 1, arr.shape[0]):
            if np.isfinite(arr[i]) and np.isfinite(out[i - 1]):
                out[i] = (out[i - 1] * (n - 1) + arr[i]) / n
            else:
                out[i] = np.nan
        return pd.Series(out, index=s.index)

    def _lwma(s: pd.Series, n: int) -> pd.Series:
        if n <= 1:
            return s.copy()
        w = np.arange(1, n + 1, dtype=float)
        denom = w.sum()
        return s.rolling(n, min_periods=n).apply(
            lambda x: np.dot(x, w) / denom if np.isfinite(x).all() else np.nan, raw=True
        )

    def _ma_on_array(s: pd.Series, n: int, m: int) -> pd.Series:
        if m == 0:
            return _sma(s, n)
        elif m == 1:
            return _ema(s, n)
        elif m == 2:
            return _smma(s, n)
        elif m == 3:
            return _lwma(s, n)
        else:
            return _sma(s, n)

    if smoothing:
        ma = _ma_on_array(buff, int(max(1, period)), int(mode))
    else:
        ma = buff.copy()

    cond_up = ma > ma.shift(1)
    up = ma.where(cond_up, np.nan)
    down = ma.where(~cond_up, np.nan)

    out = pd.DataFrame(
        {
            'Jp': ma.astype(float),
            'JpUp': up.astype(float),
            'JpDown': down.astype(float),
        },
        index=df.index,
    )
    return out