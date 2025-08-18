import numpy as np
import pandas as pd

def EhlersTwoPoleSuperSmootherFilter(df: pd.DataFrame, cutoff_period: int = 15) -> pd.Series:
    """Return the Two-Pole Super Smoother Filter (Ehlers) as a Series aligned to df.index.
    - Uses Open price (as in the original MQL4 code).
    - Handles NaNs by restarting the recursion on each contiguous valid segment.
    - Preserves length with NaNs where inputs are NaN.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(out, index=df.index, name=f"EhlersTwoPoleSuperSmootherFilter_{cutoff_period}")

    # Validate/prepare parameters
    p = float(cutoff_period)
    if not np.isfinite(p) or p <= 0:
        return pd.Series(out, index=df.index, name=f"EhlersTwoPoleSuperSmootherFilter_{cutoff_period}")

    # Coefficients per Ehlers' formulation (using sqrt(2) instead of 1.414)
    a1 = np.exp(-np.sqrt(2.0) * np.pi / p)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / p)
    coef3 = -(a1 * a1)
    coef2 = b1
    coef1 = 1.0 - coef2 - coef3

    x = df["Open"].to_numpy(dtype=float, copy=False)
    valid = np.isfinite(x)

    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        # Find end of this contiguous valid segment
        j = i
        while j < n and valid[j]:
            j += 1
        seg_len = j - i

        # Warmup: first three values of the segment equal to price (as in original MQL)
        if seg_len >= 1:
            out[i] = x[i]
        if seg_len >= 2:
            out[i + 1] = x[i + 1]
        if seg_len >= 3:
            out[i + 2] = x[i + 2]

        # Recurrence from the 4th element of the segment onward
        for k in range(i + 3, j):
            out[k] = coef1 * x[k] + coef2 * out[k - 1] + coef3 * out[k - 2]

        i = j

    return pd.Series(out, index=df.index, name=f"EhlersTwoPoleSuperSmootherFilter_{cutoff_period}")