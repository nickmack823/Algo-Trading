import numpy as np
import pandas as pd

def Smoothstep(df: pd.DataFrame, period: int = 32, price: str = "close") -> pd.DataFrame:
    """
    SmoothStep indicator (mladen, 2022) converted to pandas/numpy.

    Parameters
    - df: DataFrame with ['Timestamp','Open','High','Low','Close','Volume']
    - period: rolling window length (default 32, coerced to >=1)
    - price: one of {'close','open','high','low','median','typical','weighted','lowhigh'}

    Returns
    - DataFrame with columns:
        'SmoothStep'        : main smoothstep line
        'SmoothStepDownA'   : alternating down-trend overlay segment A
        'SmoothStepDownB'   : alternating down-trend overlay segment B
      Aligned to df.index with NaNs for warmup.
    """
    period = max(int(period), 1)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    p = c.copy()
    ph = p.copy()
    pl = p.copy()

    mode = str(price).lower()
    if mode == "open":
        p = ph = pl = o
    elif mode == "high":
        p = ph = pl = h
    elif mode == "low":
        p = ph = pl = l
    elif mode == "median":
        p = ph = pl = (h + l) / 2.0
    elif mode == "typical":
        p = ph = pl = (h + l + c) / 3.0
    elif mode == "weighted":
        p = ph = pl = (h + l + 2.0 * c) / 4.0
    elif mode == "lowhigh":
        p = c
        ph = h
        pl = l
    else:  # "close" or default
        p = ph = pl = c

    low_roll = pl.rolling(window=period, min_periods=period).min()
    high_roll = ph.rolling(window=period, min_periods=period).max()
    denom = high_roll - low_roll

    raw = (p - low_roll) / denom
    # If denom == 0 and not NaN -> 0; if denom is NaN -> NaN
    raw = raw.where(denom != 0, 0.0)

    val = raw * raw * (3.0 - 2.0 * raw)

    delta = val.diff()
    valc = pd.Series(np.where(delta > 0, 1.0, np.where(delta < 0, 2.0, np.nan)), index=df.index)
    # Do not carry trend through NaN warmup; ffill afterwards
    valc = valc.ffill()

    down_mask = (valc == 2.0) & val.notna()

    run_start = down_mask & (~down_mask.shift(1, fill_value=False))
    rid = run_start.cumsum()
    odd = (rid % 2 == 1)

    a_mask = down_mask & odd
    b_mask = down_mask & (~odd)

    # Include previous bar at the start of each down run (to mimic iPlotPoint connectivity)
    a_mask = a_mask | (run_start & odd).shift(-1, fill_value=False)
    b_mask = b_mask | (run_start & (~odd)).shift(-1, fill_value=False)

    out = pd.DataFrame(index=df.index)
    out["SmoothStep"] = val
    out["SmoothStepDownA"] = val.where(a_mask)
    out["SmoothStepDownB"] = val.where(b_mask)
    return out