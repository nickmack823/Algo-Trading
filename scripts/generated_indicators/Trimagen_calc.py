import numpy as np
import pandas as pd

def Trimagen(df: pd.DataFrame, period: int = 20, applied_price: str = 'close') -> pd.Series:
    """Return the primary indicator line(s) aligned to df.index; vectorized, handle NaNs, stable defaults. For the other parameters, use what is specifically required for this specific indicator."""
    p = int(period) if period is not None else 20
    if p < 1:
        p = 1

    len1 = int(np.floor((p + 1.0) / 2.0))
    len2 = int(np.ceil((p + 1.0) / 2.0))

    ap = (applied_price or 'close').lower()
    if ap == 'close':
        price = df['Close']
    elif ap == 'open':
        price = df['Open']
    elif ap == 'high':
        price = df['High']
    elif ap == 'low':
        price = df['Low']
    elif ap == 'median':
        price = (df['High'] + df['Low']) / 2.0
    elif ap == 'typical':
        price = (df['High'] + df['Low'] + df['Close']) / 3.0
    elif ap == 'weighted':
        price = (df['High'] + df['Low'] + 2.0 * df['Close']) / 4.0
    else:
        price = df['Close']

    sma1 = price.rolling(window=len1, min_periods=len1).mean()
    trimagen = sma1.rolling(window=len2, min_periods=len2).mean()
    trimagen.name = 'TriMAgen'
    return trimagen