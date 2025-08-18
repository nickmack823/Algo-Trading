import numpy as np
import pandas as pd

def T3MA(df: pd.DataFrame, length: int = 10, b: float = 0.88, price: int = 0) -> pd.Series:
    """Return the T3 moving average as a pandas Series aligned to df.index; vectorized, handles NaNs, stable defaults.
    
    Parameters:
    - length: smoothing length (int > 0), default 10
    - b: T3 'b' coefficient (float), default 0.88
    - price: applied price selector (int), default 0
        0=Close, 1=Open, 2=High, 3=Low, 4=Median(HL2), 5=Typical(HLC3), 6=Weighted(HLCC4)
    """
    if length <= 0:
        raise ValueError("length must be a positive integer")
    cols = df.columns
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(cols)):
        raise ValueError(f"df must contain columns: {sorted(required)}")

    if price == 0:
        pr = df["Close"]
    elif price == 1:
        pr = df["Open"]
    elif price == 2:
        pr = df["High"]
    elif price == 3:
        pr = df["Low"]
    elif price == 4:
        pr = (df["High"] + df["Low"]) / 2.0
    elif price == 5:
        pr = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif price == 6:
        pr = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        raise ValueError("price must be an int in {0,1,2,3,4,5,6}")

    w1 = 4.0 / (3.0 + float(length))

    e1 = pr.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e2 = e1.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e3 = e2.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e4 = e3.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e5 = e4.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e6 = e5.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()

    b2 = b * b
    b3 = b2 * b
    c1 = -b3
    c2 = 3.0 * (b2 + b3)
    c3 = -3.0 * (2.0 * b2 + b + b3)
    c4 = 1.0 + 3.0 * b + b3 + 3.0 * b2

    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    t3.name = "T3MA"
    return t3