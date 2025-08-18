import numpy as np
import pandas as pd

def Hlctrend(
    df: pd.DataFrame,
    close_period: int = 5,
    low_period: int = 13,
    high_period: int = 34
) -> pd.DataFrame:
    """Return ALL lines as columns ['first','second']; aligned to df.index.
    first = EMA(Close, close_period) - EMA(High, high_period)
    second = EMA(Low, low_period) - EMA(Close, close_period)
    Uses EMA with smoothing 2/(n+1); warmup NaNs via min_periods per leg.
    """
    close = df["Close"].astype(float)
    low = df["Low"].astype(float)
    high = df["High"].astype(float)

    emac = close.ewm(span=close_period, adjust=False, min_periods=close_period).mean()
    emal = low.ewm(span=low_period, adjust=False, min_periods=low_period).mean()
    emah = high.ewm(span=high_period, adjust=False, min_periods=high_period).mean()

    first = emac - emah
    second = emal - emac

    return pd.DataFrame({"first": first, "second": second}, index=df.index)