import numpy as np
import pandas as pd

def Forecast(df: pd.DataFrame, length: int = 20, price: int = 0) -> pd.Series:
    """Return the Forecast Oscillator (FO) as a percent: 100*(Price - TSF)/Price.
    TSF is computed from a rolling linear regression on the previous `length` bars (excluding current),
    following the original MQL4 implementation (tsf = a + b)."""
    if length < 1:
        raise ValueError("length must be >= 1")
    if price not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError("price must be in {0,1,2,3,4,5,6}")

    # Applied price mapping (MQL4 PRICE_*)
    if price == 0:
        s = df["Close"].astype(float)
    elif price == 1:
        s = df["Open"].astype(float)
    elif price == 2:
        s = df["High"].astype(float)
    elif price == 3:
        s = df["Low"].astype(float)
    elif price == 4:  # Median
        s = ((df["High"] + df["Low"]) / 2.0).astype(float)
    elif price == 5:  # Typical
        s = ((df["High"] + df["Low"] + df["Close"]) / 3.0).astype(float)
    else:  # price == 6, Weighted (HLCC/4)
        s = ((df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0).astype(float)

    N = len(s)
    L = int(length)

    # Previous L values (exclude current bar)
    y = s.shift(1).to_numpy()

    # Prepare cumulative sums for efficient rolling weighted sums
    idx = np.arange(1, N + 1, dtype=float)  # 1-based positional index
    y0 = np.where(np.isnan(y), 0.0, y)

    cs_y = np.cumsum(y0)
    cs_i_y = np.cumsum(y0 * idx)
    cs_cnt = np.cumsum((~np.isnan(y)).astype(int))

    # Pad with leading zero for windowed differences
    cs_y = np.concatenate(([0.0], cs_y))
    cs_i_y = np.concatenate(([0.0], cs_i_y))
    cs_cnt = np.concatenate(([0], cs_cnt))

    sum_y = np.full(N, np.nan)
    sum_i_y = np.full(N, np.nan)
    cnt = np.zeros(N, dtype=float)

    valid_idx = np.arange(N) >= (L - 1)
    if valid_idx.any():
        t1 = np.where(valid_idx)[0] + 1  # 1-based end index of window
        t0 = t1 - L

        sum_y[valid_idx] = cs_y[t1] - cs_y[t0]
        sum_i_y[valid_idx] = cs_i_y[t1] - cs_i_y[t0]
        cnt[valid_idx] = cs_cnt[t1] - cs_cnt[t0]

    # Compute sxy with weights 1..L for the last L values (most recent has weight 1)
    sxy = np.full(N, np.nan)
    if valid_idx.any():
        sxy[valid_idx] = t1.astype(float) * sum_y[valid_idx] - sum_i_y[valid_idx]

    # Constants for x = 1..L
    sx = L * (L + 1) / 2.0
    sx2 = L * (L + 1) * (2 * L + 1) / 6.0
    den = (L * sx2 - sx * sx)

    # Linear regression parameters and TSF=a+b (per original MQL4 code)
    if L == 1 or den == 0.0:
        # With a single point, use last value as forecast
        tsf = sum_y  # equals the last y in the window
    else:
        b = (L * sxy - sx * sum_y) / den
        a = (sum_y - b * sx) / L
        tsf = a + b

    pr = s.to_numpy()
    fo = np.full(N, np.nan)
    mask_full = (cnt == L) & np.isfinite(tsf) & np.isfinite(pr) & (pr != 0.0)
    fo[mask_full] = 100.0 * (pr[mask_full] - tsf[mask_full]) / pr[mask_full]

    return pd.Series(fo, index=df.index, name="FO")