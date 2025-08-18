import numpy as np
import pandas as pd

def BamsBung3(
    df: pd.DataFrame,
    length: int = 14,
    deviation: float = 2.0,
    money_risk: float = 0.02,
    signal_mode: int = 1,  # 1: signals & stops, 0: only stops, 2: only signals
    line_mode: int = 1     # 1: show line, 0: hide line
) -> pd.DataFrame:
    """Return all Bams Bung 3 outputs as columns aligned to df.index.
    - Uses SMA-based Bollinger Bands on Close with ddof=0 for std.
    - Sequential logic replicates the original MQL4 algorithm.
    - Preserves length; NaNs at warmup (pre-length).
    """
    close = pd.to_numeric(df['Close'], errors='coerce')
    n = len(close)

    sma = close.rolling(length, min_periods=length).mean()
    std = close.rolling(length, min_periods=length).std(ddof=0)
    upper = sma + deviation * std
    lower = sma - deviation * std

    smax = upper.to_numpy(dtype=float)
    smin = lower.to_numpy(dtype=float)

    smax_adj = np.copy(smax)
    smin_adj = np.copy(smin)
    bsmax = np.full(n, np.nan, dtype=float)
    bsmin = np.full(n, np.nan, dtype=float)
    trend = np.zeros(n, dtype=int)

    up_stop = np.full(n, np.nan, dtype=float)
    down_stop = np.full(n, np.nan, dtype=float)
    up_signal = np.full(n, np.nan, dtype=float)
    down_signal = np.full(n, np.nan, dtype=float)
    up_line = np.full(n, np.nan, dtype=float)
    down_line = np.full(n, np.nan, dtype=float)

    c = close.to_numpy(dtype=float)

    for t in range(n):
        if t == 0 or np.isnan(smax[t]) or np.isnan(smin[t]) or np.isnan(c[t]) or np.isnan(smax_adj[t-1]) or np.isnan(smin_adj[t-1]):
            # Warmup or insufficient history
            if t > 0:
                trend[t] = trend[t-1]
            continue

        tr = trend[t-1]

        if c[t] > smax_adj[t-1]:
            tr = 1
        if c[t] < smin_adj[t-1]:
            tr = -1

        # Adjust smin/smax based on trend persistence
        smin_t = smin[t]
        smax_t = smax[t]
        if tr > 0:
            smin_t = np.maximum(smin_t, smin_adj[t-1])
        if tr < 0:
            smax_t = np.minimum(smax_t, smax_adj[t-1])

        smin_adj[t] = smin_t
        smax_adj[t] = smax_t

        rng = smax_t - smin_t
        if np.isfinite(rng):
            bsmax_t = smax_t + 0.5 * (money_risk - 1.0) * rng
            bsmin_t = smin_t - 0.5 * (money_risk - 1.0) * rng
            if tr > 0 and np.isfinite(bsmin[t-1]):
                bsmin_t = np.maximum(bsmin_t, bsmin[t-1])
            if tr < 0 and np.isfinite(bsmax[t-1]):
                bsmax_t = np.minimum(bsmax_t, bsmax[t-1])
            bsmax[t] = bsmax_t
            bsmin[t] = bsmin_t

        trend[t] = tr

        if tr > 0 and np.isfinite(bsmin[t]):
            # Up trend
            prev_up_inactive = (t > 0 and up_stop[t-1] == -1.0)
            if signal_mode > 0 and prev_up_inactive:
                up_signal[t] = bsmin[t]
                up_stop[t] = bsmin[t]
                if line_mode > 0:
                    up_line[t] = bsmin[t]
            else:
                up_stop[t] = bsmin[t]
                if line_mode > 0:
                    up_line[t] = bsmin[t]
                up_signal[t] = -1.0

            if signal_mode == 2:
                up_stop[t] = 0.0

            down_signal[t] = -1.0
            down_stop[t] = -1.0
            down_line[t] = np.nan

        elif tr < 0 and np.isfinite(bsmax[t]):
            # Down trend
            prev_down_inactive = (t > 0 and down_stop[t-1] == -1.0)
            if signal_mode > 0 and prev_down_inactive:
                down_signal[t] = bsmax[t]
                down_stop[t] = bsmax[t]
                if line_mode > 0:
                    down_line[t] = bsmax[t]
            else:
                down_stop[t] = bsmax[t]
                if line_mode > 0:
                    down_line[t] = bsmax[t]
                down_signal[t] = -1.0

            if signal_mode == 2:
                down_stop[t] = 0.0

            up_signal[t] = -1.0
            up_stop[t] = -1.0
            up_line[t] = np.nan
        # else: trend == 0; leave NaNs (warmup or no established trend)

    out = pd.DataFrame({
        'UpTrendStop': up_stop,
        'DownTrendStop': down_stop,
        'UpTrendSignal': up_signal,
        'DownTrendSignal': down_signal,
        'UpTrendLine': up_line,
        'DownTrendLine': down_line
    }, index=df.index)

    return out