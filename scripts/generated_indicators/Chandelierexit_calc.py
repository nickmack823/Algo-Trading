import numpy as np
import pandas as pd

def Chandelierexit(
    df: pd.DataFrame,
    lookback: int = 7,
    atr_period: int = 9,
    atr_mult: float = 2.5,
    shift: int = 0
) -> pd.DataFrame:
    """Return Chandelier Exit long/short lines as columns aligned to df.index.
    Parameters:
      - lookback: window for highest high / lowest low
      - atr_period: ATR period (Wilder)
      - atr_mult: ATR multiplier
      - shift: bars to offset the window/ATR into the past
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame({"Chandelier_Long": [], "Chandelier_Short": []}, index=df.index)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # ATR (Wilder's smoothing via EWM alpha=1/period; NaN warmup handled by min_periods)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()

    # Shift inputs per MQL logic (use values from 'shift' bars ago)
    high_s = high.shift(shift)
    low_s = low.shift(shift)
    atr_s = atr.shift(shift)

    # Raw stop levels
    hh = high_s.rolling(window=lookback, min_periods=lookback).max()
    ll = low_s.rolling(window=lookback, min_periods=lookback).min()

    raw_up = hh - atr_s * atr_mult
    raw_dn = ll + atr_s * atr_mult

    # Allocate outputs
    long_line = np.full(n, np.nan, dtype=float)
    short_line = np.full(n, np.nan, dtype=float)
    direction = np.zeros(n, dtype=int)

    ru = raw_up.to_numpy()
    rd = raw_dn.to_numpy()
    c = close.to_numpy()

    for j in range(n):
        prev_dir = direction[j - 1] if j > 0 else 0
        prev_ru = ru[j - 1] if j > 0 else np.nan
        prev_rd = rd[j - 1] if j > 0 else np.nan

        d = prev_dir
        if not np.isnan(c[j]):
            if not np.isnan(prev_rd) and c[j] > prev_rd:
                d = 1
            if not np.isnan(prev_ru) and c[j] < prev_ru:
                d = -1
        direction[j] = d

        up = ru[j]
        dn = rd[j]

        if d > 0:
            if j > 0 and not np.isnan(prev_ru):
                if np.isnan(up) or up < prev_ru:
                    up = prev_ru
            long_line[j] = up
            short_line[j] = np.nan
        elif d < 0:
            if j > 0 and not np.isnan(prev_rd):
                if np.isnan(dn) or dn > prev_rd:
                    dn = prev_rd
            short_line[j] = dn
            long_line[j] = np.nan
        else:
            long_line[j] = np.nan
            short_line[j] = np.nan

    return pd.DataFrame(
        {
            "Chandelier_Long": long_line,
            "Chandelier_Short": short_line
        },
        index=df.index
    )