import numpy as np
import pandas as pd

def MomentumCandlesModifiedWAtr(df: pd.DataFrame, atr_period: int = 50, atr_multiplier: float = 2.5) -> pd.DataFrame:
    """Return Value and threshold lines as columns aligned to df.index; vectorized, handle NaNs.
    
    Parameters:
    - atr_period: ATR lookback period (default 50 as in the MQL4 code)
    - atr_multiplier: threshold multiplier (default 2.5 as in the MQL4 code)
    """
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/float(atr_period), adjust=False, min_periods=atr_period).mean()

    value = (close - open_) / atr.replace(0.0, np.nan)

    thr_pos_val = 1.0 / float(atr_multiplier)
    thr_neg_val = -thr_pos_val

    valid_mask = atr.notna()
    threshold_pos = pd.Series(np.where(valid_mask, thr_pos_val, np.nan), index=df.index, dtype=float)
    threshold_neg = pd.Series(np.where(valid_mask, thr_neg_val, np.nan), index=df.index, dtype=float)

    out = pd.DataFrame({
        'Value': value.astype(float),
        'Threshold_Pos': threshold_pos,
        'Threshold_Neg': threshold_neg
    }, index=df.index)

    return out