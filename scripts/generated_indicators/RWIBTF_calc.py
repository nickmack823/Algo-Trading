import numpy as np
import pandas as pd
from typing import Optional

def RWIBTF(df: pd.DataFrame, length: int = 2, tf: Optional[str] = None) -> pd.DataFrame:
    """Random Walk Index (BTF-capable).
    
    Parameters:
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume']
    - length: maximum lookback i for RWI (i from 1..length); default 2
    - tf: optional pandas offset alias (e.g., '15T','1H','1D') to compute on higher timeframe
          and broadcast back to original index. If None, uses current dataframe timeframe.
    
    Returns:
    - DataFrame with columns ['RWIH','RWIL','TR'], aligned to df.index, NaNs for warmup.
    """
    length = int(max(1, length))

    def _compute_rwi(ohlc: pd.DataFrame, length: int) -> pd.DataFrame:
        high = pd.to_numeric(ohlc['High'], errors='coerce')
        low = pd.to_numeric(ohlc['Low'], errors='coerce')
        close = pd.to_numeric(ohlc['Close'], errors='coerce')

        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (prev_close - low).abs()))

        h_cols = []
        l_cols = []
        for i in range(1, length + 1):
            atr_i = tr.rolling(window=i, min_periods=i).mean() / np.sqrt(i + 1)
            atr_i = atr_i.replace(0.0, np.nan)

            num_h = high - low.shift(i)
            num_l = high.shift(i) - low

            ratio_h = num_h / atr_i
            ratio_l = num_l / atr_i

            h_cols.append(ratio_h)
            l_cols.append(ratio_l)

        rwi_h = pd.concat(h_cols, axis=1).max(axis=1, skipna=True).clip(lower=0)
        rwi_l = pd.concat(l_cols, axis=1).max(axis=1, skipna=True).clip(lower=0)

        out = pd.DataFrame({'RWIH': rwi_h, 'RWIL': rwi_l, 'TR': tr}, index=ohlc.index)
        return out

    if tf is None:
        return _compute_rwi(df[['High', 'Low', 'Close']], length).reindex(df.index)

    # Multi-timeframe path
    ts = pd.to_datetime(df['Timestamp'], errors='coerce')
    # Resample to higher timeframe using OHLCV convention
    ohlc_htf = (
        df.set_index(ts)
          .resample(tf, label='left', closed='left')
          .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    )

    rwi_htf = _compute_rwi(ohlc_htf[['High', 'Low', 'Close']], length)

    # Broadcast back to original index by mapping each timestamp to its period start
    period_keys = ts.dt.floor(tf)
    mapped = rwi_htf.reindex(period_keys.values)
    mapped.index = df.index
    return mapped[['RWIH', 'RWIL', 'TR']]