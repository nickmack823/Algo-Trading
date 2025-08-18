import numpy as np
import pandas as pd

def ISCalculation(df: pd.DataFrame, period: int = 10, nbchandelier: int = 10, lag: int = 0) -> pd.DataFrame:
    """Return ADJASUROPPO ('Pente') and its EMA trigger as columns aligned to df.index.
    Parameters:
      - period: EMA period for base EMA and trigger base
      - nbchandelier: lookback distance for slope (difference divided by nbchandelier)
      - lag: added to period for the trigger EMA period
    """
    close = df['Close'].astype(float)

    p = max(int(period), 1)
    n = max(int(nbchandelier), 1)
    p_trig = max(p + int(lag), 1)

    ema = close.ewm(span=p, adjust=False, min_periods=p).mean()
    pente = (ema - ema.shift(n)) / n
    trigger = pente.ewm(span=p_trig, adjust=False, min_periods=p_trig).mean()

    out = pd.DataFrame({
        'Pente': pente,
        'Trigger': trigger
    }, index=df.index)

    return out