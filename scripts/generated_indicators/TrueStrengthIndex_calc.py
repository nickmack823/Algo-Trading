import numpy as np
import pandas as pd

def TrueStrengthIndex(df: pd.DataFrame, first_r: int = 5, second_s: int = 8) -> pd.Series:
    """Return the True Strength Index (TSI) as a pandas Series aligned to df.index.
    Uses nested EMAs of momentum and its absolute value:
        TSI = 100 * EMA(EMA(mtm, first_r), second_s) / EMA(EMA(|mtm|, first_r), second_s)
    Defaults match the provided MQL4 script. Vectorized, NaNs preserved during warmup.
    """
    if first_r <= 0 or second_s <= 0:
        raise ValueError("first_r and second_s must be positive integers.")

    close = df["Close"].astype(float)

    mtm = close.diff()  # Close - Close.shift(1)
    abs_mtm = mtm.abs()

    ema_mtm = mtm.ewm(span=first_r, adjust=False, min_periods=first_r).mean()
    ema_abs_mtm = abs_mtm.ewm(span=first_r, adjust=False, min_periods=first_r).mean()

    ema2_mtm = ema_mtm.ewm(span=second_s, adjust=False, min_periods=second_s).mean()
    ema2_abs_mtm = ema_abs_mtm.ewm(span=second_s, adjust=False, min_periods=second_s).mean()

    tsi_values = np.divide(
        100.0 * ema2_mtm,
        ema2_abs_mtm,
        out=np.full_like(ema2_mtm, np.nan),
        where=ema2_abs_mtm != 0,
    )

    tsi = pd.Series(tsi_values, index=df.index, name=f"TSI_{first_r}_{second_s}")
    return tsi