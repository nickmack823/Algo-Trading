import numpy as np
import pandas as pd

def CyberCycle(df: pd.DataFrame, alpha: float = 0.07, price: str | int = 'hl2') -> pd.DataFrame:
    """
    Cyber Cycle (Ehlers) with Trigger.
    Returns a DataFrame with ['Cycle','Trigger'], aligned to df.index, preserving length with NaNs for warmup.

    Parameters:
    - alpha: float (default 0.07)
    - price: one of {'close','open','high','low','hl2','median','typical','hlc3','ohlc4','wclose','hlcc4'} or MT4 codes {0..6}
             0: Close, 1: Open, 2: High, 3: Low, 4: (H+L)/2, 5: (H+L+C)/3, 6: (H+L+2C)/4
    """
    o = df['Open'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)

    # Map price selection
    if isinstance(price, int):
        code = price
        if code == 0:
            s = c
        elif code == 1:
            s = o
        elif code == 2:
            s = h
        elif code == 3:
            s = l
        elif code == 4:
            s = (h + l) / 2.0
        elif code == 5:
            s = (h + l + c) / 3.0
        elif code == 6:
            s = (h + l + 2.0 * c) / 4.0
        else:
            raise ValueError("Unsupported MT4 price code. Use 0..6.")
    else:
        key = str(price).lower()
        if key in ('close', 'c'):
            s = c
        elif key in ('open', 'o'):
            s = o
        elif key in ('high', 'h'):
            s = h
        elif key in ('low', 'l'):
            s = l
        elif key in ('hl2', 'median'):
            s = (h + l) / 2.0
        elif key in ('typical', 'hlc3'):
            s = (h + l + c) / 3.0
        elif key in ('ohlc4',):
            s = (o + h + l + c) / 4.0
        elif key in ('wclose', 'hlcc4', 'weighted'):
            s = (h + l + 2.0 * c) / 4.0
        else:
            raise ValueError("Unsupported price string.")

    # Smooth = (Price + 2*Price[1] + 2*Price[2] + Price[3]) / 6
    smooth = (s + 2.0 * s.shift(1) + 2.0 * s.shift(2) + s.shift(3)) / 6.0

    # Precompute second difference of smoothed price
    d_smooth = smooth - 2.0 * smooth.shift(1) + smooth.shift(2)

    n = len(df)
    cycle = np.full(n, np.nan, dtype=float)

    s_vals = s.to_numpy(dtype=float)
    d_vals = d_smooth.to_numpy(dtype=float)

    c0 = (1.0 - 0.5 * alpha) ** 2
    b1 = 2.0 * (1.0 - alpha)
    b2 = - (1.0 - alpha) ** 2

    # Seed first bars as per Ehlers: for currentbar < 7 use raw price second difference / 4
    # Cycle_t = (Price_t - 2*Price_{t-1} + Price_{t-2}) / 4
    for t in range(n):
        if t < 7:
            if t >= 2 and np.isfinite(s_vals[t]) and np.isfinite(s_vals[t-1]) and np.isfinite(s_vals[t-2]):
                cycle[t] = (s_vals[t] - 2.0 * s_vals[t-1] + s_vals[t-2]) / 4.0
            else:
                cycle[t] = np.nan
        else:
            # Full recursive filter
            if t >= 2 and np.isfinite(d_vals[t]) and np.isfinite(cycle[t-1]) and np.isfinite(cycle[t-2]):
                cycle[t] = c0 * d_vals[t] + b1 * cycle[t-1] + b2 * cycle[t-2]
            else:
                cycle[t] = np.nan

    cycle_s = pd.Series(cycle, index=df.index, name='Cycle')
    trigger_s = cycle_s.shift(1).rename('Trigger')

    return pd.DataFrame({'Cycle': cycle_s, 'Trigger': trigger_s})