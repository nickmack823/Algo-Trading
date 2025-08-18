import numpy as np
import pandas as pd

def TTF(
    df: pd.DataFrame,
    ttf_bars: int = 8,
    top_line: float = 75.0,
    bottom_line: float = -75.0,
    t3_period: int = 3,
    b: float = 0.7,
) -> pd.DataFrame:
    """Trend Trigger Factor (TTF) with T3 smoothing.
    Returns a DataFrame with columns ['TTF','Signal'] aligned to df.index.
    Vectorized using pandas/numpy, NaNs for warmup.
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=["TTF", "Signal"], dtype=float)

    n = max(int(ttf_bars), 1)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Recent and older rolling extrema
    hh_recent = high.rolling(window=n, min_periods=n).max()
    ll_recent = low.rolling(window=n, min_periods=n).min()
    hh_older = hh_recent.shift(n)
    ll_older = ll_recent.shift(n)

    buy_power = hh_recent - ll_older
    sell_power = hh_older - ll_recent

    den = 0.5 * (buy_power + sell_power)
    den = den.replace(0.0, np.nan)
    ttf_raw = (buy_power - sell_power) / den * 100.0

    # T3 smoothing parameters
    r = float(max(t3_period, 1))
    r = 1.0 + 0.5 * (r - 1.0)  # as in the MQL4 code
    alpha = 2.0 / (r + 1.0)

    b2 = b * b
    b3 = b2 * b
    c1 = -b3
    c2 = 3.0 * (b2 + b3)
    c3 = -3.0 * (2.0 * b2 + b + b3)
    c4 = 1.0 + 3.0 * b + b3 + 3.0 * b2

    def ema_zero_seed(x: pd.Series, a: float) -> pd.Series:
        """EMA with initial previous value = 0 (zero-seeded), vectorized via ewm."""
        y = x.copy()
        fv = y.first_valid_index()
        if fv is not None:
            y.loc[fv] = y.loc[fv] * a  # ensures y_fv = a * x_fv (zero seed)
        return y.ewm(alpha=a, adjust=False, ignore_na=True).mean()

    e1 = ema_zero_seed(ttf_raw, alpha)
    e2 = ema_zero_seed(e1, alpha)
    e3 = ema_zero_seed(e2, alpha)
    e4 = ema_zero_seed(e3, alpha)
    e5 = ema_zero_seed(e4, alpha)
    e6 = ema_zero_seed(e5, alpha)

    ttf_main = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    signal = pd.Series(np.where(ttf_main >= 0.0, top_line, bottom_line), index=df.index, dtype=float)
    signal = signal.where(ttf_main.notna())

    return pd.DataFrame({"TTF": ttf_main, "Signal": signal}, index=df.index)