import numpy as np
import pandas as pd

def DorseyInertia(df: pd.DataFrame, rvi_period: int = 10, avg_period: int = 14, smoothing_period: int = 20) -> pd.Series:
    """Dorsey Inertia indicator (Mladen implementation).
    
    Parameters
    - rvi_period: int = 10, period for rolling std of High/Low (population std, SMA-based)
    - avg_period: int = 14, EMA period (alpha=1/avg_period) for averaging up/down std components
    - smoothing_period: int = 20, final SMA period of the combined RVI
    
    Returns
    - pd.Series named 'Inertia', aligned to df.index with NaNs for warmup
    """
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    # Rolling standard deviations (population, SMA-based)
    std_high = high.rolling(window=int(rvi_period), min_periods=int(rvi_period)).std(ddof=0)
    std_low = low.rolling(window=int(rvi_period), min_periods=int(rvi_period)).std(ddof=0)

    # Up/Down moves
    dh = high.diff()
    dl = low.diff()
    up_h = dh > 0
    dn_h = dh < 0
    up_l = dl > 0
    dn_l = dl < 0

    # Build u/d components with NaNs prior to sufficient std history
    stdh_vals = std_high.values
    stdl_vals = std_low.values
    valid_h = ~np.isnan(stdh_vals)
    valid_l = ~np.isnan(stdl_vals)

    u_high = np.where(valid_h, np.where(up_h.fillna(False).values, stdh_vals, 0.0), np.nan)
    d_high = np.where(valid_h, np.where(dn_h.fillna(False).values, stdh_vals, 0.0), np.nan)
    u_low = np.where(valid_l, np.where(up_l.fillna(False).values, stdl_vals, 0.0), np.nan)
    d_low = np.where(valid_l, np.where(dn_l.fillna(False).values, stdl_vals, 0.0), np.nan)

    u_high = pd.Series(u_high, index=df.index)
    d_high = pd.Series(d_high, index=df.index)
    u_low = pd.Series(u_low, index=df.index)
    d_low = pd.Series(d_low, index=df.index)

    # EMA smoothing with alpha=1/avg_period (matches ((p-1)*prev + x)/p)
    alpha = 1.0 / float(avg_period)
    HUp = u_high.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    HDo = d_high.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    LUp = u_low.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    LDo = d_low.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()

    # RVI components (guard division by zero; preserve NaNs for warmup)
    denom_h = HUp + HDo
    denom_l = LUp + LDo
    rvih = (100.0 * HUp / denom_h)
    rvil = (100.0 * LUp / denom_l)
    rvih = rvih.mask(denom_h == 0.0, 0.0)
    rvil = rvil.mask(denom_l == 0.0, 0.0)

    rvi = (rvih + rvil) / 2.0

    # Final SMA smoothing
    inertia = rvi.rolling(window=int(smoothing_period), min_periods=int(smoothing_period)).mean()
    return inertia.rename('Inertia')