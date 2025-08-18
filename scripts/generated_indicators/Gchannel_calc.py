import numpy as np
import pandas as pd

def Gchannel(df: pd.DataFrame, length: int = 100, price: str = 'close') -> pd.DataFrame:
    """Return all G-Channel lines as columns ['Upper','Middle','Lower'], aligned to df.index.
    
    Params:
    - length: int >= 1 (default 100)
    - price: one of {'close','open','high','low','median','typical','weighted'} (default 'close')
    
    Notes:
    - Recursive by definition; runs in O(n) with a single forward pass.
    - NaNs in source are handled by carrying forward the last computed values; leading NaNs remain NaN.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(index=df.index, columns=['Upper', 'Middle', 'Lower'], dtype=float)

    length = int(max(1, length))
    p = str(price).lower()

    c = df['Close'].to_numpy(dtype=float, copy=False)
    h = df['High'].to_numpy(dtype=float, copy=False)
    l = df['Low'].to_numpy(dtype=float, copy=False)
    o = df['Open'].to_numpy(dtype=float, copy=False)

    if p == 'close':
        src = c.copy()
    elif p == 'open':
        src = o.copy()
    elif p == 'high':
        src = h.copy()
    elif p == 'low':
        src = l.copy()
    elif p == 'median':
        src = (h + l) / 2.0
    elif p == 'typical':
        src = (h + l + c) / 3.0
    elif p == 'weighted':
        src = (h + l + 2.0 * c) / 4.0
    else:
        raise ValueError("price must be one of {'close','open','high','low','median','typical','weighted'}")

    a = np.full(n, np.nan, dtype=float)
    b = np.full(n, np.nan, dtype=float)

    prev_a = 0.0
    prev_b = 0.0
    have_prev = False  # track if we've computed at least one valid step

    for i in range(n):
        x = src[i]
        if not np.isfinite(x):
            if i > 0:
                # carry forward last computed values if available
                a[i] = a[i - 1]
                b[i] = b[i - 1]
            # do not update prev_a/prev_b on NaN
            continue

        ap = prev_a if have_prev else 0.0
        bp = prev_b if have_prev else 0.0

        ai = max(x, ap) - (ap - bp) / length
        bi = min(x, bp) + (ap - bp) / length

        a[i] = ai
        b[i] = bi
        prev_a = ai
        prev_b = bi
        have_prev = True

    mid = (a + b) / 2.0

    out = pd.DataFrame(
        {
            'Upper': a,
            'Middle': mid,
            'Lower': b,
        },
        index=df.index,
    )
    return out