import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

def WprMaAlerts(
    df: pd.DataFrame,
    wpr_period: int = 35,
    signal_period: int = 21,
    ma_method: str = "smma",
) -> pd.DataFrame:
    """Williams %R with signal MA and cross state.

    Parameters:
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume']
    - wpr_period: int, lookback for Williams %R (default 35)
    - signal_period: int, lookback for signal MA (default 21)
    - ma_method: str, one of {'sma','ema','smma','lwma'} (default 'smma')

    Returns:
    - DataFrame with columns ['WPR','Signal','Cross'], aligned to df.index
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # Williams %R
    hh = high.rolling(window=int(max(1, wpr_period)), min_periods=1).max()
    ll = low.rolling(window=int(max(1, wpr_period)), min_periods=1).min()
    denom = (hh - ll).to_numpy()
    wpr_vals = np.where(denom != 0, -100.0 * (hh.to_numpy() - close.to_numpy()) / denom, 0.0)
    wpr = pd.Series(wpr_vals, index=df.index)

    # Signal MA helpers
    def _lwma(series: pd.Series, period: int) -> pd.Series:
        period = max(1, int(period))
        s = series.to_numpy(dtype=float)
        n = s.shape[0]
        out = np.full(n, np.nan, dtype=float)

        # Early ramp: variable window sizes 1..min(period-1, n)
        early = min(period - 1, n)
        for t in range(early):
            w = t + 1
            window = s[:w]
            mask = ~np.isnan(window)
            if mask.any():
                wts = np.arange(1, w + 1, dtype=float)[mask]
                out[t] = np.dot(window[mask], wts) / wts.sum()

        # Steady state: full window = period
        if n >= period:
            sw = sliding_window_view(s, window_shape=period)  # shape (n-period+1, period)
            mask = ~np.isnan(sw)
            wts = np.arange(1, period + 1, dtype=float)
            denom = (wts * mask).sum(axis=1)
            num = np.nansum(sw * wts, axis=1)
            vals = num / denom
            out[period - 1 :] = vals

        return pd.Series(out, index=series.index)

    mm = str(ma_method).strip().lower()
    p = max(1, int(signal_period))
    if p <= 1:
        sig = wpr.copy()
    elif mm == "sma":
        sig = wpr.rolling(window=p, min_periods=1).mean()
    elif mm == "ema":
        sig = wpr.ewm(alpha=2.0 / (p + 1.0), adjust=False, min_periods=1).mean()
    elif mm == "smma":
        sig = wpr.ewm(alpha=1.0 / p, adjust=False, min_periods=1).mean()
    elif mm == "lwma":
        sig = _lwma(wpr, p)
    else:
        # Fallback: no smoothing
        sig = wpr.copy()

    # Cross state: 1 if WPR > Signal, -1 if WPR < Signal, else carry previous; NaN where diff is NaN
    diff = wpr - sig
    raw = np.sign(diff.to_numpy())
    raw[np.isnan(diff.to_numpy())] = np.nan
    cross = pd.Series(raw, index=df.index)
    cross_filled = cross.replace(0.0, np.nan).ffill()
    cross_final = cross_filled.where(~diff.isna(), np.nan).fillna(0.0).astype(float)
    # Cast to int where not NaN; keep NaN as NaN
    cross_int = pd.Series(np.where(cross_final.notna(), cross_final.astype(int), np.nan), index=df.index)

    out = pd.DataFrame(
        {
            "WPR": wpr,
            "Signal": sig,
            "Cross": cross_int,
        },
        index=df.index,
    )
    return out