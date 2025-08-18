import numpy as np
import pandas as pd

def Silence(df: pd.DataFrame, my_period: int = 12, buff_size: int = 96, point: float = 0.0001, redraw: bool = True) -> pd.DataFrame:
    """Silence indicator (Aggressiveness, Volatility) scaled 0..100 (reversed), aligned to df.index.
    - my_period: lookback for bar calculations (default 12)
    - buff_size: lookback for normalization window (default 96)
    - point: instrument tick size (default 0.0001)
    - redraw: if False, last bar equals previous (freeze current bar)
    """
    o = df['Open'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)

    # Aggressiveness raw: rolling sum over my_period of sign(candle) * (close - prev_close)
    sgn = np.where(c > o, 1.0, -1.0)
    dclose = c.diff()
    b_contrib = pd.Series(sgn, index=df.index) * dclose
    aggress_raw = b_contrib.rolling(window=my_period, min_periods=my_period).sum() / (point * my_period)

    # Volatility raw: (highest high - lowest low) over my_period
    hh = h.rolling(window=my_period, min_periods=my_period).max()
    ll = l.rolling(window=my_period, min_periods=my_period).min()
    vol_raw = (hh - ll) / (point * my_period)

    # Normalization over last buff_size values (map [MIN,MAX] -> [100,0])
    def normalize(raw: pd.Series) -> pd.Series:
        roll_max = raw.rolling(window=buff_size, min_periods=buff_size).max()
        roll_min = raw.rolling(window=buff_size, min_periods=buff_size).min()
        denom = roll_max - roll_min
        out = 100.0 * (roll_max - raw) / denom
        out = out.where(denom != 0, 1e7)  # match MQL behavior when range == 0
        return out

    aggress = normalize(aggress_raw)
    vol = normalize(vol_raw)

    out = pd.DataFrame(
        {
            'Aggressiveness': aggress,
            'Volatility': vol,
        },
        index=df.index,
    )

    # Freeze last bar if redraw is False
    if not redraw and len(out) >= 2:
        out.iloc[-1] = out.iloc[-2]

    return out