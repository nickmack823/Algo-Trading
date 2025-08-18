import numpy as np
import pandas as pd

def ErgodicTVI(
    df: pd.DataFrame,
    Period1: int = 12,
    Period2: int = 12,
    Period3: int = 1,
    EPeriod1: int = 5,
    EPeriod2: int = 5,
    EPeriod3: int = 5,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """Return ETVI and Signal lines as columns aligned to df.index.
    Uses MQL4-style EMA (seeded with SMA of first 'period' values). Vectorized; preserves NaNs for warmup.
    """
    def _ema_mql(s: pd.Series, period: int) -> pd.Series:
        s = s.astype(float)
        if period is None or period <= 1:
            return s.copy()
        sma = s.rolling(window=period, min_periods=period).mean()
        arr = s.to_numpy(dtype=float)
        sma_arr = sma.to_numpy(dtype=float)
        idxs = np.flatnonzero(~np.isnan(sma_arr))
        if idxs.size == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        first = idxs[0]
        arr2 = arr.copy()
        arr2[:first] = np.nan
        arr2[first] = sma_arr[first]
        ema = pd.Series(arr2, index=s.index).ewm(span=period, adjust=False, min_periods=1).mean()
        return ema

    if df.empty:
        return pd.DataFrame({"ETVI": pd.Series(dtype=float), "Signal": pd.Series(dtype=float)}, index=df.index)

    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    upticks = (v + (c - o) / float(pip_size)) / 2.0
    dnticks = v - upticks

    ema_up1 = _ema_mql(upticks, Period1)
    ema_dn1 = _ema_mql(dnticks, Period1)
    ema_up2 = _ema_mql(ema_up1, Period2)
    ema_dn2 = _ema_mql(ema_dn1, Period2)

    denom = ema_up2 + ema_dn2
    tv = 100.0 * (ema_up2 - ema_dn2) / denom
    tv = tv.mask(denom == 0.0, 0.0)

    tvi = _ema_mql(tv, Period3)
    ema_tvi1 = _ema_mql(tvi, EPeriod1)
    etvi = _ema_mql(ema_tvi1, EPeriod2)
    signal = _ema_mql(etvi, EPeriod3)

    return pd.DataFrame({"ETVI": etvi, "Signal": signal}, index=df.index)