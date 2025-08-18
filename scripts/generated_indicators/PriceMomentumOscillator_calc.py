import numpy as np
import pandas as pd

def PriceMomentumOscillator(df: pd.DataFrame, one: int = 35, two: int = 20, period: int = 10) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Two-stage custom EMA smoothing (alpha=2/one, 2/two) of Close percentage change,
    then EMA signal with period='period' on PMO."""
    if one <= 0 or two <= 0 or period <= 0:
        raise ValueError("Parameters 'one', 'two', and 'period' must be positive integers.")

    close = df['Close'].astype(float)

    raw_one = (close / close.shift(1) * 100.0) - 100.0

    alpha1 = 2.0 / float(one)
    alpha2 = 2.0 / float(two)

    c1 = raw_one.ewm(alpha=alpha1, adjust=False, min_periods=1).mean()
    raw_two = 10.0 * c1
    pmo = raw_two.ewm(alpha=alpha2, adjust=False, min_periods=1).mean()

    signal = pmo.ewm(span=period, adjust=False, min_periods=1).mean()

    out = pd.DataFrame({
        'PMO': pmo,
        'Signal': signal
    }, index=df.index)

    return out