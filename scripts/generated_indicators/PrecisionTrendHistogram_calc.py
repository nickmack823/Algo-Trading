import numpy as np
import pandas as pd

def PrecisionTrendHistogram(df: pd.DataFrame, avg_period: int = 30, sensitivity: float = 3.0) -> pd.DataFrame:
    """Precision Trend Histogram (Up/Down histograms and internal Trend state).
    Returns a DataFrame with columns ['Up','Down','Trend'] aligned to df.index.
    - avg_period: rolling average period for range (High-Low)
    - sensitivity: multiplier applied to the rolling average
    Vectorized where possible; sequential state update as per original MQL4 logic."""
    if df.empty:
        return pd.DataFrame(index=df.index, columns=["Up", "Down", "Trend"], dtype=float)

    high = df["High"].to_numpy(dtype=float, copy=False)
    low = df["Low"].to_numpy(dtype=float, copy=False)
    close = df["Close"].to_numpy(dtype=float, copy=False)

    period = max(int(avg_period), 1)

    rng = high - low
    avg = pd.Series(rng, index=df.index).rolling(window=period, min_periods=1).mean().to_numpy() * float(sensitivity)

    n = len(df)
    trend = np.full(n, np.nan, dtype=float)
    avgd = np.full(n, np.nan, dtype=float)
    avgu = np.full(n, np.nan, dtype=float)
    minc = np.full(n, np.nan, dtype=float)
    maxc = np.full(n, np.nan, dtype=float)

    valid = np.isfinite(close) & np.isfinite(avg)
    if not np.any(valid):
        return pd.DataFrame({"Up": np.full(n, np.nan), "Down": np.full(n, np.nan), "Trend": trend}, index=df.index)

    # Initialize at first valid index
    k = int(np.argmax(valid))
    if valid[k]:
        trend[k] = 0.0
        avgd[k] = close[k] - avg[k]
        avgu[k] = close[k] + avg[k]
        minc[k] = close[k]
        maxc[k] = close[k]

    last = k
    # Sequential state updates
    for t in range(k + 1, n):
        if not valid[t]:
            continue

        # Carry forward previous state
        trend[t] = trend[last]
        avgd[t] = avgd[last]
        avgu[t] = avgu[last]
        minc[t] = minc[last]
        maxc[t] = maxc[last]

        tp = trend[last]

        if tp == 0:
            if close[t] > avgu[last]:
                minc[t] = close[t]
                avgd[t] = close[t] - avg[t]
                trend[t] = 1.0
            if close[t] < avgd[last]:
                maxc[t] = close[t]
                avgu[t] = close[t] + avg[t]
                trend[t] = -1.0

        elif tp == 1:
            avgd[t] = minc[last] - avg[t]
            if close[t] > minc[last]:
                minc[t] = close[t]
            if close[t] < avgd[last]:
                maxc[t] = close[t]
                avgu[t] = close[t] + avg[t]
                trend[t] = -1.0

        elif tp == -1:
            avgu[t] = maxc[last] + avg[t]
            if close[t] < maxc[last]:
                maxc[t] = close[t]
            if close[t] > avgu[last]:
                minc[t] = close[t]
                avgd[t] = close[t] - avg[t]
                trend[t] = 1.0

        last = t

    up = np.where(trend == 1.0, 1.0, np.nan)
    down = np.where(trend == -1.0, 1.0, np.nan)

    return pd.DataFrame({"Up": up, "Down": down, "Trend": trend}, index=df.index)