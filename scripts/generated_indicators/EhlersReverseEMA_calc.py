import numpy as np
import pandas as pd

def EhlersReverseEMA(df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Parameters:
      - alpha: smoothing factor in (0,1], default 0.1.
    """
    close = df['Close'].astype(float)
    alpha = float(alpha)
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    delta = 1.0 - alpha

    # Reverse-time EMA (anti-causal): reverse, ewm, reverse back
    ema = close.iloc[::-1].ewm(alpha=alpha, adjust=False).mean().iloc[::-1]

    # Cascaded reverse components
    re = delta * ema + ema.shift(-1)  # re1
    exp = 2
    for _ in range(7):  # re2 .. re8 with exponents 2,4,8,16,32,64,128
        re = (delta ** exp) * re + re.shift(-1)
        exp *= 2

    main = ema - alpha * re

    # Warmup NaNs (75 bars as per original implementation note)
    warmup = min(75, len(main))
    if warmup > 0:
        main.iloc[:warmup] = np.nan
        ema.iloc[:warmup] = np.nan

    out = pd.DataFrame({'Main': main, 'EMA': ema}, index=df.index)
    return out