import numpy as np
import pandas as pd

def CMO(df: pd.DataFrame, length: int = 9, price: str = "Close") -> pd.Series:
    """
    Chande Momentum Oscillator (CMO)
    Returns a pandas Series aligned to df.index.
    Parameters:
    - length: lookback period (default 9)
    - price: applied price, one of ['Close','Open','High','Low','Median','Typical','Weighted'] (default 'Close')
    """
    if length is None or int(length) < 1:
        return pd.Series(np.nan, index=df.index, name="CMO")
    length = int(length)

    p = price.lower()
    if p == "close":
        ap = df["Close"]
    elif p == "open":
        ap = df["Open"]
    elif p == "high":
        ap = df["High"]
    elif p == "low":
        ap = df["Low"]
    elif p == "median":
        ap = (df["High"] + df["Low"]) / 2.0
    elif p == "typical":
        ap = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif p == "weighted":
        ap = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        ap = df["Close"]

    diff = ap - ap.shift(1)
    gains = diff.clip(lower=0)
    losses = (-diff).clip(lower=0)

    s1 = gains.rolling(window=length, min_periods=length).mean()
    s2 = losses.rolling(window=length, min_periods=length).mean()
    denom = s1 + s2

    cmo = np.where(denom != 0, (s1 - s2) / denom * 100.0, np.nan)
    out = pd.Series(cmo, index=df.index, name="CMO")
    return out