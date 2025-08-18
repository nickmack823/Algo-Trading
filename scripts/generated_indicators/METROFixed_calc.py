import numpy as np
import pandas as pd

def METROFixed(
    df: pd.DataFrame,
    period_rsi: int = 14,
    step_size_fast: float = 5.0,
    step_size_slow: float = 15.0
) -> pd.DataFrame:
    """Return RSI and StepRSI (fast/slow) aligned to df.index.
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume']
    - period_rsi: RSI period (Wilder's smoothing)
    - step_size_fast: step size for fast StepRSI (in RSI points)
    - step_size_slow: step size for slow StepRSI (in RSI points)
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame({"RSI": [], "StepRSI_fast": [], "StepRSI_slow": []}, index=df.index)

    close = df["Close"].astype(float)

    # Wilder's RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period_rsi, adjust=False, min_periods=period_rsi).mean()
    avg_loss = loss.ewm(alpha=1.0 / period_rsi, adjust=False, min_periods=period_rsi).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.clip(lower=0.0, upper=100.0)

    rsi_vals = rsi.to_numpy(dtype=float)

    step_fast = np.full(n, np.nan, dtype=float)
    step_slow = np.full(n, np.nan, dtype=float)

    # Initialize with last available RSI if finite
    if np.isfinite(rsi_vals[-1]):
        step_fast[-1] = rsi_vals[-1]
        step_slow[-1] = rsi_vals[-1]

        # Backward recursion (from end to start), clamped by step sizes
        for i in range(n - 2, -1, -1):
            r = rsi_vals[i]
            nf = step_fast[i + 1]
            ns = step_slow[i + 1]

            # Fast
            lo_f = r - step_size_fast
            hi_f = r + step_size_fast
            step_fast[i] = np.minimum(np.maximum(nf, lo_f), hi_f)

            # Slow
            lo_s = r - step_size_slow
            hi_s = r + step_size_slow
            step_slow[i] = np.minimum(np.maximum(ns, lo_s), hi_s)

    out = pd.DataFrame(
        {
            "RSI": rsi,
            "StepRSI_fast": step_fast,
            "StepRSI_slow": step_slow,
        },
        index=df.index,
    )
    return out