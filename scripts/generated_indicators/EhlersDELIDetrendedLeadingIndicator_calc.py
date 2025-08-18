import numpy as np
import pandas as pd

def EhlersDELIDetrendedLeadingIndicator(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the DELI primary indicator line aligned to df.index; vectorized, handles NaNs."""
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    # Stepwise extremes per original logic
    cond_high = (high > high.shift(1))
    prevhigh = pd.Series(np.where(cond_high, high, np.nan), index=df.index)
    prevhigh.iloc[0] = high.iloc[0] if high.notna().iloc[0] else np.nan
    prevhigh = prevhigh.ffill()

    cond_low = (low < low.shift(1))
    prevlow = pd.Series(np.where(cond_low, low, np.nan), index=df.index)
    prevlow.iloc[0] = low.iloc[0] if low.notna().iloc[0] else np.nan
    prevlow = prevlow.ffill()

    price = (prevhigh + prevlow) / 2.0

    p = max(int(period), 1)
    alpha = 2.0 / (p + 1.0)
    alpha2 = alpha / 2.0

    ema1 = price.ewm(alpha=alpha, adjust=False).mean()
    ema2 = price.ewm(alpha=alpha2, adjust=False).mean()
    dsp = ema1 - ema2
    temp = dsp.ewm(alpha=alpha, adjust=False).mean()
    deli = dsp - temp
    deli.name = 'DELI'

    return deli