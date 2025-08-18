import numpy as np
import pandas as pd

def TopBottomNR(df: pd.DataFrame, per: int = 14) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index. Vectorized, handles NaNs, stable defaults."""
    if per < 1:
        raise ValueError("per must be >= 1")

    low = df['Low']
    high = df['High']

    prev_low_min = low.shift(1).rolling(window=per, min_periods=per).min()
    prev_high_max = high.shift(1).rolling(window=per, min_periods=per).max()

    valid_long = prev_low_min.notna()
    valid_short = prev_high_max.notna()

    reset_long = (low < prev_low_min).fillna(False)
    reset_short = (high > prev_high_max).fillna(False)

    def _run_length(reset: pd.Series, valid: pd.Series) -> pd.Series:
        n = len(reset)
        idx = np.arange(n, dtype=np.int64)
        reset_arr = reset.to_numpy(dtype=bool)
        valid_arr = valid.to_numpy(dtype=bool)
        boundary = reset_arr | (~valid_arr)
        last_boundary = np.maximum.accumulate(np.where(boundary, idx, -1))
        res = (idx - last_boundary).astype(float)
        res[~valid_arr] = np.nan
        return pd.Series(res, index=reset.index)

    long_signal = _run_length(reset_long, valid_long)
    short_signal = _run_length(reset_short, valid_short)

    return pd.DataFrame({
        'LongSignal': long_signal,
        'ShortSignal': short_signal
    }, index=df.index)