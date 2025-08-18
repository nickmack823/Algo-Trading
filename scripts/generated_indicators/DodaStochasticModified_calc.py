import numpy as np
import pandas as pd

def DodaStochasticModified(
    df: pd.DataFrame,
    Slw: int = 8,
    Pds: int = 13,
    Slwsignal: int = 9
) -> pd.DataFrame:
    """
    Doda-Stochastic (modified) based on MT4 source:
    - EmaClose: EMA(Close, Slw)
    - StochOfEma: 0-100 stochastic of EmaClose over Pds window
    - DodaStoch: EMA(StochOfEma, Slw)
    - DodaSignal: EMA(DodaStoch, Slwsignal)

    Returns a DataFrame with all lines aligned to df.index and NaNs during warmup.
    """
    Slw = max(int(Slw), 1)
    Pds = max(int(Pds), 1)
    Slwsignal = max(int(Slwsignal), 1)

    close = df["Close"].astype(float)

    ema_close = close.ewm(span=Slw, adjust=False, min_periods=Slw).mean()

    roll_min = ema_close.rolling(window=Pds, min_periods=Pds).min()
    roll_max = ema_close.rolling(window=Pds, min_periods=Pds).max()
    denom = roll_max - roll_min

    mask = denom != 0
    stoch_of_ema = pd.Series(np.nan, index=df.index, dtype=float)
    stoch_of_ema.loc[mask] = 100.0 * (ema_close.loc[mask] - roll_min.loc[mask]) / denom.loc[mask]

    doda_stoch = stoch_of_ema.ewm(span=Slw, adjust=False, min_periods=Slw).mean()
    doda_signal = doda_stoch.ewm(span=Slwsignal, adjust=False, min_periods=Slwsignal).mean()

    out = pd.DataFrame(
        {
            "DodaStoch": doda_stoch,
            "DodaSignal": doda_signal,
            "EmaClose": ema_close,
            "StochOfEma": stoch_of_ema,
        },
        index=df.index,
    )
    return out