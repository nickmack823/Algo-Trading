import numpy as np
import pandas as pd

def RangeFilterModified(df: pd.DataFrame, atr_period: int = 14, multiplier: float = 3.0) -> pd.DataFrame:
    """Return ALL lines as columns ['LineCenter','LineUp','LineDn'] aligned to df.index.
    - Uses Wilder ATR(atr_period) * multiplier as offset.
    - Fully vectorized where possible; O(n) recursion for the center line by necessity.
    - Preserves length with NaNs during warmup.
    """
    n = len(df)
    idx = df.index

    if n == 0 or atr_period <= 0:
        return pd.DataFrame({'LineCenter': pd.Series(index=idx, dtype=float),
                             'LineUp': pd.Series(index=idx, dtype=float),
                             'LineDn': pd.Series(index=idx, dtype=float)})

    h = df['High'].to_numpy(dtype=float)
    l = df['Low'].to_numpy(dtype=float)
    c = df['Close'].to_numpy(dtype=float)

    # True Range
    prev_c = np.empty_like(c)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]
    tr = np.nanmax(np.vstack([
        h - l,
        np.abs(h - prev_c),
        np.abs(l - prev_c)
    ]), axis=0)

    # Wilder's RMA for ATR
    def _rma(x: np.ndarray, period: int) -> np.ndarray:
        y = np.full_like(x, np.nan, dtype=float)
        if x.size == 0 or period <= 0:
            return y
        # determine first seed index as SMA over first 'period' values (or first window with full data)
        if np.all(np.isfinite(x[:period])):
            seed_idx = period - 1
            y[seed_idx] = x[:period].mean()
        else:
            roll_mean = pd.Series(x).rolling(period, min_periods=period).mean().values
            finite_idx = np.flatnonzero(np.isfinite(roll_mean))
            if finite_idx.size == 0:
                return y
            seed_idx = finite_idx[0]
            y[seed_idx] = roll_mean[seed_idx]
        alpha = 1.0 / period
        for i in range(seed_idx + 1, x.size):
            xi = x[i]
            if np.isfinite(xi):
                yi_1 = y[i - 1]
                if np.isfinite(yi_1):
                    y[i] = yi_1 + (xi - yi_1) * alpha
                else:
                    y[i] = xi
            else:
                y[i] = y[i - 1]
        return y

    atr = _rma(tr, atr_period)
    smooth = atr * float(multiplier)

    lc = np.full(n, np.nan, dtype=float)
    up = np.full(n, np.nan, dtype=float)
    dn = np.full(n, np.nan, dtype=float)

    # find first index where we can seed the center line
    valid = np.isfinite(smooth) & np.isfinite(c)
    if not np.any(valid):
        return pd.DataFrame({'LineCenter': pd.Series(lc, index=idx),
                             'LineUp': pd.Series(up, index=idx),
                             'LineDn': pd.Series(dn, index=idx)})

    i0 = int(np.flatnonzero(valid)[0])
    lc[i0] = c[i0]  # seed with close; Up/Down remain NaN at seed to respect dependency on prior center

    for i in range(i0 + 1, n):
        prev = lc[i - 1]
        si = smooth[i]
        ci = c[i]

        if not np.isfinite(prev) or not np.isfinite(si) or not np.isfinite(ci):
            lc[i] = prev
            continue

        up[i] = prev + si
        dn[i] = prev - si

        lci = prev
        if ci > up[i]:
            lci = ci - si
        elif ci < dn[i]:
            lci = ci + si
        lc[i] = lci

    return pd.DataFrame({'LineCenter': pd.Series(lc, index=idx),
                         'LineUp': pd.Series(up, index=idx),
                         'LineDn': pd.Series(dn, index=idx)})