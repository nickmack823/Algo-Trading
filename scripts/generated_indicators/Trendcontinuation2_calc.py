import numpy as np
import pandas as pd

def Trendcontinuation2(df: pd.DataFrame, n: int = 20, t3_period: int = 5, b: float = 0.618) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Vectorized implementation of 'Trend continuation factor 2' (MT4) using pandas only.
    Params:
      - n: lookback length for change aggregation (default 20). Effective window is n+1.
      - t3_period: T3 smoothing period (default 5).
      - b: T3 smoothing factor (default 0.618).
    """
    close = df["Close"].astype(float)

    # 1) One-bar changes split into positive/negative components (strict >0 / <0 as in MQL4)
    delta = close.diff()
    pos = delta.where(delta > 0, 0.0)             # positive changes; zeros elsewhere; NaNs preserved
    neg = (-delta).where(delta < 0, 0.0)          # magnitude of negative changes; zeros elsewhere

    # 2) CF_p / CF_n: cumulative sums within consecutive runs of pos/neg, resetting on non-pos/non-neg
    mask_pos = pos > 0
    grp_pos = (~mask_pos).cumsum()
    CF_p = pos.fillna(0.0).groupby(grp_pos).cumsum()
    CF_p = CF_p.where(mask_pos, 0.0)

    mask_neg = neg > 0
    grp_neg = (~mask_neg).cumsum()
    CF_n = neg.fillna(0.0).groupby(grp_neg).cumsum()
    CF_n = CF_n.where(mask_neg, 0.0)

    # 3) Rolling sums over [i-n, i] => window length n+1
    win = max(int(n), 0) + 1
    ch_p = pos.rolling(window=win, min_periods=win).sum()
    ch_n = neg.rolling(window=win, min_periods=win).sum()
    cff_p = CF_p.rolling(window=win, min_periods=win).sum()
    cff_n = CF_n.rolling(window=win, min_periods=win).sum()

    k_p = ch_p - cff_n
    k_n = ch_n - cff_p

    # 4) T3 smoothing (Tillson T3) with MetaTrader-style alpha adjustment
    b2 = b * b
    b3 = b2 * b
    c1 = -b3
    c2 = 3.0 * (b2 + b3)
    c3 = -3.0 * (2.0 * b2 + b + b3)
    c4 = 1.0 + 3.0 * b + b3 + 3.0 * b2

    n1 = max(int(t3_period), 1)
    n1 = 1.0 + 0.5 * (n1 - 1.0)
    alpha = 2.0 / (n1 + 1.0)

    def t3_filter(x: pd.Series) -> pd.Series:
        e1 = x.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e2 = e1.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e3 = e2.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e4 = e3.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e5 = e4.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e6 = e5.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    out_pos = t3_filter(k_p)
    out_neg = t3_filter(k_n)

    return pd.DataFrame(
        {
            "TrendContinuation2_Pos": out_pos,
            "TrendContinuation2_Neg": out_neg,
        },
        index=df.index,
    )