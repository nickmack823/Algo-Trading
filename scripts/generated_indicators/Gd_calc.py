import numpy as np
import pandas as pd

def Gd(df: pd.DataFrame, length: int = 20, vf: float = 0.7, price: int = 0) -> pd.DataFrame:
    """Return ALL lines as columns ['GD','EMA'] aligned to df.index. Vectorized, handles NaNs, and preserves warmup NaNs.
    
    Params:
    - length: EMA period (int >= 1)
    - vf: generalization factor (float)
    - price: applied price code:
        0=Close, 1=Open, 2=High, 3=Low, 4=Median(HL2), 5=Typical(HLC3), 6=Weighted(HLCC4)
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    # Applied price selection
    if price == 0:
        src = df["Close"]
    elif price == 1:
        src = df["Open"]
    elif price == 2:
        src = df["High"]
    elif price == 3:
        src = df["Low"]
    elif price == 4:
        src = (df["High"] + df["Low"]) / 2.0
    elif price == 5:
        src = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif price == 6:
        src = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        src = df["Close"]

    alpha = 2.0 / (length + 1.0)

    ema = src.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    ema_of_ema = ema.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    gd = (1.0 + vf) * ema - vf * ema_of_ema

    out = pd.DataFrame({"GD": gd, "EMA": ema}, index=df.index)
    return out