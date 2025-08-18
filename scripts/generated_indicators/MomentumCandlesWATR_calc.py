import numpy as np
import pandas as pd

def MomentumCandlesWATR(df: pd.DataFrame, atr_period: int = 50, atr_multiplier: float = 2.5) -> pd.DataFrame:
    """Return Bull/Bear Open/Close buffers for Momentum Candles with ATR filter.
    Signals:
      - Bullish if Close > Open and ATR(atr_period)/abs(Close-Open) < atr_multiplier
      - Bearish if Close < Open and ATR(atr_period)/abs(Close-Open) < atr_multiplier
    Outputs are aligned to df.index with NaNs where no signal.
    """
    o = df['Open'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)

    # True Range
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

    # Wilder's ATR via EWM(alpha=1/period, adjust=False)
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    body = (c - o).abs()
    ratio = atr / body.replace(0, np.nan)

    up_mask = (c > o) & (ratio < atr_multiplier)
    dn_mask = (c < o) & (ratio < atr_multiplier)

    bull_open = np.where(up_mask, o, np.nan)
    bull_close = np.where(up_mask, c, np.nan)
    bear_open = np.where(dn_mask, o, np.nan)
    bear_close = np.where(dn_mask, c, np.nan)

    out = pd.DataFrame(
        {
            'BullOpen': bull_open,
            'BullClose': bull_close,
            'BearOpen': bear_open,
            'BearClose': bear_close,
        },
        index=df.index,
    )

    return out