import numpy as np
import pandas as pd

def DpoHistogramIndicator(df: pd.DataFrame, period: int = 14, ma: str = 'sma') -> pd.DataFrame:
    """Detrended Price Oscillator with up/down histograms.
    Returns DataFrame with columns ['DPO_Up','DPO_Dn','DPO'] aligned to df.index.
    
    Parameters:
    - period: lookback length for the moving average (default 14).
    - ma: moving average type: 'sma' (default), 'ema', 'smma' (Wilder/RMA), or 'wma' (LWMA).
    """
    if period is None or period < 1:
        raise ValueError("period must be a positive integer")

    close = pd.to_numeric(df['Close'], errors='coerce')

    ma_lower = str(ma).lower()
    if ma_lower == 'sma':
        ma_series = close.rolling(window=period, min_periods=period).mean()
    elif ma_lower == 'ema':
        ma_series = close.ewm(span=period, adjust=False, min_periods=period).mean()
    elif ma_lower in ('smma', 'rma', 'wilder'):
        # Wilder's smoothing (SMMA/RMA) is EMA with alpha = 1/period
        ma_series = close.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    elif ma_lower in ('wma', 'lwma'):
        weights = np.arange(1, period + 1, dtype=float)
        w_sum = weights.sum()
        ma_series = close.rolling(window=period, min_periods=period).apply(
            lambda x: np.dot(x, weights) / w_sum, raw=True
        )
    else:
        raise ValueError("ma must be one of: 'sma', 'ema', 'smma'/'rma'/'wilder', 'wma'/'lwma'")

    # DPO uses MA shifted forward by t_prd bars (iMA ma_shift = period//2 + 1)
    t_prd = period // 2 + 1
    shifted_ma = ma_series.shift(-t_prd)
    dpo = close - shifted_ma

    # State: 1 if dpo>0, -1 if dpo<0, zeros inherit previous state; NaNs remain NaN then filled with 0
    valc = pd.Series(np.sign(dpo.values), index=df.index)
    zero_mask = valc.eq(0)
    valc = valc.mask(zero_mask, np.nan).ffill().fillna(0.0)

    dpo_up = dpo.where(valc.eq(1.0))
    dpo_dn = dpo.where(valc.eq(-1.0))

    out = pd.DataFrame(
        {
            'DPO_Up': dpo_up.astype(float),
            'DPO_Dn': dpo_dn.astype(float),
            'DPO': dpo.astype(float),
        },
        index=df.index,
    )
    return out