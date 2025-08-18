import numpy as np
import pandas as pd

def Coral(df: pd.DataFrame, length: int = 34, coef: float = 0.4) -> pd.DataFrame:
    """THV Coral indicator (MT4) using pandas/numpy only.
    
    Returns a DataFrame with:
    - 'Coral': primary coral line
    - 'Coral_Yellow', 'Coral_RoyalBlue', 'Coral_Red': segmented lines for coloring logic (NaN where not active)
    
    Parameters:
    - length: base length (default 34)
    - coef: filter coefficient (default 0.4)
    """
    idx = df.index
    close = df['Close'].astype(float).to_numpy()

    # Ensure valid length
    n = max(int(length), 1)
    # MT4 code: g = (n - 1)/2 + 1; alpha = 2/(g + 1) => alpha = 4/(n + 3)
    alpha = 4.0 / (n + 3.0)

    c = float(coef)
    c2 = c * c
    c3 = c2 * c
    A0 = -c3
    A1 = 3.0 * (c2 + c3)
    A2 = -3.0 * (2.0 * c2 + c + c3)
    A3 = 3.0 * c + 1.0 + c3 + 3.0 * c2

    def _ema_zero_init(x: np.ndarray, a: float) -> np.ndarray:
        # Prepend a zero so the first output equals a*x0 (MT4 arrays start from zero-initialized state)
        z = np.empty(len(x) + 1, dtype=float)
        z[0] = 0.0
        z[1:] = x
        y = pd.Series(z).ewm(alpha=a, adjust=False).mean().to_numpy()
        return y[1:]

    # 6 cascaded EMAs with same alpha, each zero-initialized
    e1 = _ema_zero_init(close, alpha)
    e2 = _ema_zero_init(e1, alpha)
    e3 = _ema_zero_init(e2, alpha)
    e4 = _ema_zero_init(e3, alpha)
    e5 = _ema_zero_init(e4, alpha)
    e6 = _ema_zero_init(e5, alpha)

    coral = A0 * e6 + A1 * e5 + A2 * e4 + A3 * e3
    coral_s = pd.Series(coral, index=idx, name='Coral')

    # Segmented color buffers (following MT4 logic)
    yellow = coral_s.copy()
    blue = coral_s.copy()
    red = coral_s.copy()

    prev = coral_s.shift(1)
    down_mask = prev > coral_s
    up_mask = prev < coral_s
    flat_mask = ~(down_mask | up_mask)

    # MT4 logic:
    # if prev > curr: empty blue
    # else if prev < curr: empty red
    # else: empty yellow
    blue = blue.mask(down_mask, np.nan)
    red = red.mask(up_mask, np.nan)
    yellow = yellow.mask(flat_mask, np.nan)

    out = pd.DataFrame({
        'Coral': coral_s,
        'Coral_Yellow': yellow,
        'Coral_RoyalBlue': blue,
        'Coral_Red': red,
    }, index=idx)

    return out