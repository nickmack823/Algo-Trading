import numpy as np
import pandas as pd

def GeominMA(df: pd.DataFrame, length: int = 10, price: int = 0) -> pd.Series:
    """Geometric Mean Moving Average of the applied price.
    
    Params:
    - length: window size (>=1), default 10
    - price: applied price enum (as in MQL4):
        0 Close, 1 Open, 2 High, 3 Low, 4 Median(HL2), 5 Typical(HLC3), 6 Weighted(WC)
    
    Returns:
    - pd.Series aligned to df.index with NaNs for warmup/invalid windows.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if price not in {0, 1, 2, 3, 4, 5, 6}:
        raise ValueError("price must be one of {0,1,2,3,4,5,6}")
    
    # Select applied price
    if price == 0:
        ap = df['Close'].astype(float)
        price_name = 'Close'
    elif price == 1:
        ap = df['Open'].astype(float)
        price_name = 'Open'
    elif price == 2:
        ap = df['High'].astype(float)
        price_name = 'High'
    elif price == 3:
        ap = df['Low'].astype(float)
        price_name = 'Low'
    elif price == 4:
        ap = ((df['High'] + df['Low']) / 2.0).astype(float)
        price_name = 'Median'
    elif price == 5:
        ap = ((df['High'] + df['Low'] + df['Close']) / 3.0).astype(float)
        price_name = 'Typical'
    else:  # price == 6
        ap = ((df['High'] + df['Low'] + 2.0 * df['Close']) / 4.0).astype(float)
        price_name = 'Weighted'
    
    # Geometric mean via log to avoid overflow; require strictly positive prices in window
    ap_pos = ap.where(ap > 0.0)  # non-positive -> NaN so window becomes invalid
    log_ap = np.log(ap_pos)
    mean_log = log_ap.rolling(window=length, min_periods=length).mean()
    geom = np.exp(mean_log)
    geom.name = f"GeominMA_{length}_{price_name}"
    return geom