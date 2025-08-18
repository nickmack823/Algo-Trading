import numpy as np
import pandas as pd

def VortexIndicator(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Compute the Vortex Indicator (VI+ and VI-) over 'length' periods.
    Returns a DataFrame with columns ['VI_plus_{length}', 'VI_minus_{length}'] aligned to df.index.
    Vectorized using pandas; preserves length with NaNs for warmup periods.
    """
    high = df['High'].astype('float64')
    low = df['Low'].astype('float64')
    close = df['Close'].astype('float64')

    prev_low = low.shift(1)
    prev_high = high.shift(1)
    prev_close = close.shift(1)

    plus_vm = (high - prev_low).abs()
    minus_vm = (low - prev_high).abs()

    tr_hl = (high - low).abs()
    tr_hc = (high - prev_close).abs()
    tr_lc = (low - prev_close).abs()
    tr = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)

    sum_plus_vm = plus_vm.rolling(length, min_periods=length).sum()
    sum_minus_vm = minus_vm.rolling(length, min_periods=length).sum()
    sum_tr = tr.rolling(length, min_periods=length).sum().replace(0, np.nan)

    vi_plus = sum_plus_vm / sum_tr
    vi_minus = sum_minus_vm / sum_tr

    out = pd.DataFrame({
        f'VI_plus_{length}': vi_plus,
        f'VI_minus_{length}': vi_minus
    }, index=df.index)

    return out