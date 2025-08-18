import numpy as np
import pandas as pd

def SmoothedMomentum(
    df: pd.DataFrame,
    momentum_length: int = 12,
    use_smoothing: bool = True,
    smoothing_method: int = 0,
    smoothing_length: int = 20,
    price: int = 0
) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    
    Parameters:
    - momentum_length: int = 12
    - use_smoothing: bool = True
    - smoothing_method: int = 0  (0=SMA, 1=EMA, 2=SMMA/Wilder, 3=LWMA)
    - smoothing_length: int = 20
    - price: int = 0  (0=Close, 1=Open, 2=High, 3=Low, 4=Median, 5=Typical, 6=Weighted)
    """
    n = max(int(momentum_length), 1)
    m = max(int(smoothing_length), 1)

    # Applied price selection
    if price == 0:
        p = df['Close'].astype(float)
    elif price == 1:
        p = df['Open'].astype(float)
    elif price == 2:
        p = df['High'].astype(float)
    elif price == 3:
        p = df['Low'].astype(float)
    elif price == 4:
        p = (df['High'] + df['Low']) / 2.0
    elif price == 5:
        p = (df['High'] + df['Low'] + df['Close']) / 3.0
    elif price == 6:
        p = (df['High'] + df['Low'] + 2.0 * df['Close']) / 4.0
    else:
        p = df['Close'].astype(float)
    p = p.astype(float)

    # Momentum: 100 * price / price.shift(n); avoid div by zero
    denom = p.shift(n)
    denom = denom.where(denom != 0, np.nan)
    momentum = 100.0 * p / denom

    # Smoothing
    if use_smoothing:
        if m <= 1:
            sm = momentum.copy()
        else:
            if smoothing_method == 0:  # SMA
                sm = momentum.rolling(window=m, min_periods=m).mean()
            elif smoothing_method == 1:  # EMA
                sm = momentum.ewm(span=m, adjust=False).mean()
            elif smoothing_method == 2:  # SMMA (Wilder's RMA)
                sm = momentum.ewm(alpha=1.0 / m, adjust=False).mean()
            elif smoothing_method == 3:  # LWMA
                w = np.arange(1, m + 1, dtype=float)
                w_sum = w.sum()
                sm = momentum.rolling(window=m, min_periods=m).apply(
                    lambda a: np.dot(a, w) / w_sum, raw=True
                )
            else:  # default to SMA
                sm = momentum.rolling(window=m, min_periods=m).mean()
    else:
        sm = momentum.copy()

    out = pd.DataFrame(
        {
            'SM': sm,
            'Momentum': momentum
        },
        index=df.index
    )
    return out