import numpy as np
import pandas as pd

def WilliamVixFix(df: pd.DataFrame, period: int = 22) -> pd.Series:
    """Return the William VIX-FIX primary line aligned to df.index; vectorized, handles NaNs, stable defaults."""
    close = df['Close']
    low = df['Low']
    max_close = close.rolling(window=period, min_periods=period).max()
    wvf = 100.0 * (max_close - low) / max_close
    wvf = wvf.where(~(max_close <= 0), 0.0)
    wvf.name = f'WilliamVixFix_{period}'
    return wvf