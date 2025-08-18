import numpy as np
import pandas as pd

def ATRBasedEMAVariant1(df: pd.DataFrame, ema_fastest: float = 14.0, multiplier: float = 300.0) -> pd.DataFrame:
    """Return ALL lines as columns ['EMA_ATR_var1','EMA_Equivalent']; aligned to df.index.
    Based on ATR% (High/Low) smoothed by EMA(14), computes a dynamic EMA period and applies it to Close.
    Higher ATR -> slower EMA.
    """
    high = pd.to_numeric(df['High'], errors='coerce')
    low = pd.to_numeric(df['Low'], errors='coerce')
    close = pd.to_numeric(df['Close'], errors='coerce')

    # Avoid division by zero/negatives
    ratio = high / low.where(low > 0)

    # EMA of ratio with alpha=1/14 (equivalent to int_atr*13/14 + ratio/14)
    alpha_atr = 1.0 / 14.0
    int_atr = ratio.ewm(alpha=alpha_atr, adjust=False, min_periods=1).mean()

    ema_equiv = (((int_atr - 1.0) * multiplier + 1.0) * float(ema_fastest))

    # Variable alpha for the Close EMA
    alpha_dyn = 2.0 / (ema_equiv + 1.0)

    n = len(df)
    signal = np.full(n, np.nan, dtype=float)
    c_vals = close.to_numpy(dtype=float)
    a_vals = alpha_dyn.to_numpy(dtype=float)

    # Find first index where both alpha and close are valid
    valid_mask = np.isfinite(c_vals) & np.isfinite(a_vals)
    if valid_mask.any():
        start = int(np.argmax(valid_mask))  # first True index
        signal[start] = c_vals[start]
        for t in range(start + 1, n):
            c = c_vals[t]
            a = a_vals[t]
            if np.isfinite(c) and np.isfinite(a):
                signal[t] = signal[t - 1] * (1.0 - a) + c * a
            else:
                signal[t] = signal[t - 1]

    out = pd.DataFrame(
        {
            'EMA_ATR_var1': signal,
            'EMA_Equivalent': ema_equiv.astype(float).reindex(df.index),
        },
        index=df.index,
    )
    return out