import numpy as np
import pandas as pd

def ASO(df: pd.DataFrame, period: int = 10, mode: int = 0, bulls: bool = True, bears: bool = True) -> pd.DataFrame:
    """Average Sentiment Oscillator (ASO).
    Returns a DataFrame with columns ['ASO_Bulls','ASO_Bears'] aligned to df.index.
    Parameters:
      - period: lookback period for group metrics and SMA smoothing (default 10)
      - mode: 0 = average of intra-bar and group, 1 = intra-bar only, 2 = group only (default 0)
      - bulls/bears: enable computation for the respective line (default True)
    """
    p = max(int(period), 1)
    m = int(mode)
    if m not in (0, 1, 2):
        m = 0

    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    o = df['Open'].astype(float)
    c = df['Close'].astype(float)

    intrarange = h - l
    intrarange_safe = intrarange.copy()
    intrarange_safe[intrarange_safe == 0] = 1.0

    intrabar_bulls = (((c - l) + (h - o)) / 2.0) * 100.0 / intrarange_safe
    intrabar_bears = (((h - c) + (o - l)) / 2.0) * 100.0 / intrarange_safe

    grouplow = l.rolling(window=p, min_periods=p).min()
    grouphigh = h.rolling(window=p, min_periods=p).max()
    groupopen = o.shift(p - 1)
    grouprange = grouphigh - grouplow
    grouprange_safe = grouprange.copy()
    grouprange_safe[grouprange_safe == 0] = 1.0

    group_bulls = (((c - grouplow) + (grouphigh - groupopen)) / 2.0) * 100.0 / grouprange_safe
    group_bears = (((grouphigh - c) + (groupopen - grouplow)) / 2.0) * 100.0 / grouprange_safe

    if m == 0:
        temp_bulls = (intrabar_bulls + group_bulls) / 2.0
        temp_bears = (intrabar_bears + group_bears) / 2.0
    elif m == 1:
        temp_bulls = intrabar_bulls
        temp_bears = intrabar_bears
    else:  # m == 2
        temp_bulls = group_bulls
        temp_bears = group_bears

    aso_bulls = temp_bulls.rolling(window=p, min_periods=p).mean()
    aso_bears = temp_bears.rolling(window=p, min_periods=p).mean()

    out = pd.DataFrame({'ASO_Bulls': aso_bulls, 'ASO_Bears': aso_bears}, index=df.index)
    if not bulls:
        out['ASO_Bulls'] = np.nan
    if not bears:
        out['ASO_Bears'] = np.nan
    return out