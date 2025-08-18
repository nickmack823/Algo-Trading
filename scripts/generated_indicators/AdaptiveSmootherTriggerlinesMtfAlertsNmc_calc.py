import numpy as np
import pandas as pd

def AdaptiveSmootherTriggerlinesMtfAlertsNmc(
    df: pd.DataFrame,
    LsmaPeriod: int = 50,
    LsmaPrice: int = 0,
    AdaptPeriod: int = 21,
    MultiColor: bool = True,
) -> pd.DataFrame:
    """
    Triggerlines (adaptive smoother) converted from MQL4.
    Returns all lines aligned to df.index:
      - lsma: adaptive smoother
      - lsmaUa, lsmaUb: segmented down-slope parts for multi-color plotting
      - lwma: previous-bar lsma
      - lwmaUa, lwmaUb: segmented down-slope parts of lwma
      - lstrend, lwtrend: +1/-1 trend of lsma/lwma based on slope sign (ties forward-filled)

    Parameters:
      - LsmaPeriod: base period for smoothing (default 50)
      - LsmaPrice: price type (0 Close,1 Open,2 High,3 Low,4 Median,5 Typical,6 Weighted)
      - AdaptPeriod: adaptation lookback for std-dev/avg of std-dev (default 21)
      - MultiColor: if True, compute Ua/Ub segmented series for down-trend
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            columns=["lsma","lsmaUa","lsmaUb","lwma","lwmaUa","lwmaUb","lstrend","lwtrend"],
            index=df.index
        )

    # Map MQL4 price constants
    if LsmaPrice == 0:
        price = df["Close"].astype(float).to_numpy()
    elif LsmaPrice == 1:
        price = df["Open"].astype(float).to_numpy()
    elif LsmaPrice == 2:
        price = df["High"].astype(float).to_numpy()
    elif LsmaPrice == 3:
        price = df["Low"].astype(float).to_numpy()
    elif LsmaPrice == 4:
        price = ((df["High"] + df["Low"]) / 2.0).astype(float).to_numpy()
    elif LsmaPrice == 5:
        price = ((df["High"] + df["Low"] + df["Close"]) / 3.0).astype(float).to_numpy()
    elif LsmaPrice == 6:
        price = ((df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0).astype(float).to_numpy()
    else:
        price = df["Close"].astype(float).to_numpy()

    close = df["Close"].astype(float)

    # Adaptive period calculation
    dev = close.rolling(window=AdaptPeriod, min_periods=1).std(ddof=0)
    avg = dev.rolling(window=AdaptPeriod, min_periods=1).mean()
    dev_vals = dev.to_numpy()
    avg_vals = avg.to_numpy()

    period = np.empty(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dev_vals != 0.0, avg_vals / dev_vals, np.nan)
    # Where dev==0 or ratio nan, fallback to LsmaPeriod
    period[:] = LsmaPeriod
    mask_ratio = np.isfinite(ratio)
    period[mask_ratio & (dev_vals != 0.0)] = LsmaPeriod * ratio[mask_ratio & (dev_vals != 0.0)]
    period = np.maximum(period, 3.0)

    # iSmooth recursive filter (stateful, dynamic alpha per bar)
    alpha = 0.45 * (period - 1.0) / (0.45 * (period - 1.0) + 2.0)
    y = np.full(n, np.nan, dtype=float)

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0

    for t in range(n):
        pr = price[t]
        if not np.isfinite(pr):
            # Do not update states on NaN input; output NaN
            y[t] = np.nan
            continue

        if t <= 2:
            s0 = pr
            s1 = 0.0
            s2 = pr
            s3 = 0.0
            s4 = pr
            y[t] = pr
            continue

        a = float(alpha[t]) if np.isfinite(alpha[t]) else 0.0
        prev_s4 = s4
        prev_s3 = s3
        prev_s1 = s1

        s0 = pr + a * (s0 - pr)
        s1 = (pr - s0) * (1.0 - a) + a * prev_s1
        s2 = s0 + s1
        s3 = (s2 - prev_s4) * (1.0 - a) * (1.0 - a) + (a * a) * prev_s3
        s4 = s3 + prev_s4
        y[t] = s4

    lsma = pd.Series(y, index=df.index, name="lsma")
    lwma = lsma.shift(1)
    lwma.name = "lwma"

    # Trends based on slope sign; ties keep previous value
    def slope_trend(s: pd.Series) -> pd.Series:
        d = s.diff()
        tr = pd.Series(np.where(d > 0, 1.0, np.where(d < 0, -1.0, np.nan)), index=s.index)
        return tr.ffill()

    lstrend = slope_trend(lsma).rename("lstrend")
    lwtrend = slope_trend(lwma).rename("lwtrend")

    # MultiColor segmented down-trend series
    def split_down_runs(base: pd.Series, trend: pd.Series):
        if not MultiColor:
            return base.where(pd.Series([False]*len(base), index=base.index)), base.where(pd.Series([False]*len(base), index=base.index))
        m = (trend == -1.0)
        starts = m & (~m.shift(1, fill_value=False))
        run_id = starts.cumsum().where(m, 0).astype(int)
        parity = (run_id % 2)
        ua = base.where((run_id > 0) & (parity == 1))
        ub = base.where((run_id > 0) & (parity == 0))
        return ua, ub

    lsmaUa, lsmaUb = split_down_runs(lsma, lstrend)
    lwmaUa, lwmaUb = split_down_runs(lwma, lwtrend)

    out = pd.DataFrame(
        {
            "lsma": lsma,
            "lsmaUa": lsmaUa,
            "lsmaUb": lsmaUb,
            "lwma": lwma,
            "lwmaUa": lwmaUa,
            "lwmaUb": lwmaUb,
            "lstrend": lstrend,
            "lwtrend": lwtrend,
        },
        index=df.index,
    )
    return out