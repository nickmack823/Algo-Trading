import numpy as np
import pandas as pd

def TetherLine(df: pd.DataFrame, length: int = 55) -> pd.DataFrame:
    """Return all indicator buffers as columns aligned to df.index:
    - AboveCenter: midpoint of rolling [Highest High + Lowest Low]/2 when Close > midpoint
    - BelowCenter: midpoint when Close < midpoint
    - ArrowUp, ArrowDown: placeholders (NaN) per original script
    """
    length = max(1, int(length))
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)

    hh = high.rolling(window=length, min_periods=length).max()
    ll = low.rolling(window=length, min_periods=length).min()
    center = (hh + ll) / 2.0

    above = pd.Series(np.where(close > center, center, np.nan), index=df.index, name='AboveCenter')
    below = pd.Series(np.where(close < center, center, np.nan), index=df.index, name='BelowCenter')

    up = pd.Series(np.nan, index=df.index, name='ArrowUp')
    down = pd.Series(np.nan, index=df.index, name='ArrowDown')

    return pd.DataFrame({'AboveCenter': above, 'BelowCenter': below, 'ArrowUp': up, 'ArrowDown': down})