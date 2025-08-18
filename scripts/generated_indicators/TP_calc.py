import numpy as np
import pandas as pd

def TP(df: pd.DataFrame, length: int = 14, show_updn: bool = False) -> pd.DataFrame:
    """Advance Trend Pressure (TP), Up and Dn lines.
    - Returns DataFrame with columns ['TP','Up','Dn'] aligned to df.index.
    - Vectorized; uses rolling sums over 'length' bars.
    - Up/Dn columns are NaN if show_updn is False.
    """
    length = int(max(1, length))

    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']

    # Per-bar contributions (exclude bars where Close == Open -> 0 contribution)
    up_contrib = c - l
    dn_contrib = h - c

    eq_mask = (c == o)
    up_contrib = up_contrib.where(~eq_mask, 0.0)
    dn_contrib = dn_contrib.where(~eq_mask, 0.0)

    # Preserve NaNs if any input OHLC is NaN
    valid_mask = o.notna() & h.notna() & l.notna() & c.notna()
    up_contrib = up_contrib.where(valid_mask, np.nan)
    dn_contrib = dn_contrib.where(valid_mask, np.nan)

    up = up_contrib.rolling(window=length, min_periods=length).sum()
    dn = dn_contrib.rolling(window=length, min_periods=length).sum()
    tp = up - dn

    if not show_updn:
        up_out = pd.Series(np.nan, index=df.index)
        dn_out = pd.Series(np.nan, index=df.index)
    else:
        up_out = up
        dn_out = dn

    return pd.DataFrame({'TP': tp, 'Up': up_out, 'Dn': dn_out}, index=df.index)