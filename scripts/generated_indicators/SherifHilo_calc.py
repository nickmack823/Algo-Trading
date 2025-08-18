import numpy as np
import pandas as pd

def SherifHilo(df: pd.DataFrame, period_high: int = 100, period_lows: int = 100) -> pd.DataFrame:
    """Sherif HiLo indicator.
    Returns a DataFrame with columns: ['LineUp','LineDown','Data','Max','Min'] aligned to df.index.
    - period_high: rolling lookback for HHV of High
    - period_lows: rolling lookback for LLV of Low
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=['LineUp', 'LineDown', 'Data', 'Max', 'Min'], dtype=float)

    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float).to_numpy()

    # Rolling extrema (past window including current bar)
    llv_s = low.rolling(window=period_lows, min_periods=period_lows).min()
    hhv_s = high.rolling(window=period_high, min_periods=period_high).max()

    llv = llv_s.to_numpy()
    hhv = hhv_s.to_numpy()

    n = len(df)
    data = np.full(n, np.nan, dtype=float)
    lineup = np.full(n, np.nan, dtype=float)
    linedown = np.full(n, np.nan, dtype=float)

    start = max(period_high, period_lows) - 1
    prev_default = 0.0  # mimic MT4 buffers' default initialization

    for t in range(start, n):
        prev = data[t - 1] if t > 0 and np.isfinite(data[t - 1]) else prev_default

        # Default carry-over if no direction change
        data[t] = prev

        c = close[t]
        # Compare and switch regime
        if c > prev:
            data[t] = llv[t]
            lineup[t] = llv[t]
            linedown[t] = np.nan
        elif c < prev:
            data[t] = hhv[t]
            linedown[t] = hhv[t]
            lineup[t] = np.nan
        else:
            # equal: keep previous data; no line on this bar
            lineup[t] = np.nan
            linedown[t] = np.nan

    out = pd.DataFrame(
        {
            'LineUp': lineup,
            'LineDown': linedown,
            'Data': data,
            'Max': hhv,
            'Min': llv,
        },
        index=df.index,
    )
    return out