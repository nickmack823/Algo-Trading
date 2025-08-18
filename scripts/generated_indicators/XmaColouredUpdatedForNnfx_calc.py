import numpy as np
import pandas as pd

def XmaColouredUpdatedForNnfx(
    df: pd.DataFrame,
    period: int = 12,
    porog: int = 3,
    metod: str | int = "ema",
    metod2: str | int = "ema",
    price: str = "close",
    tick_size: float = 0.0001,
) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Parameters:
      - period: MA length (default 12)
      - porog: threshold in 'points' (integer), multiplied by tick_size (default 3)
      - metod/metod2: MA methods ('sma','ema','smma','lwma' or 0/1/2/3 respectively; default 'ema')
      - price: applied price ('close','open','high','low','median','typical','weighted'; default 'close')
      - tick_size: price per point (default 0.0001)
    """

    def _applied_price(data: pd.DataFrame, p: str) -> pd.Series:
        p = str(p).lower()
        if p == "close":
            return data["Close"].astype(float)
        if p == "open":
            return data["Open"].astype(float)
        if p == "high":
            return data["High"].astype(float)
        if p == "low":
            return data["Low"].astype(float)
        if p == "median":
            return (data["High"] + data["Low"]) / 2.0
        if p == "typical":
            return (data["High"] + data["Low"] + data["Close"]) / 3.0
        if p == "weighted":
            return (data["High"] + data["Low"] + 2.0 * data["Close"]) / 4.0
        # default to close
        return data["Close"].astype(float)

    def _normalize_method(m):
        m_map = {
            0: "sma",
            1: "ema",
            2: "smma",
            3: "lwma",
            "sma": "sma",
            "ema": "ema",
            "smma": "smma",
            "rma": "smma",
            "lwma": "lwma",
            "wma": "lwma",
        }
        key = m if isinstance(m, (int, np.integer)) else str(m).lower()
        return m_map.get(key, "ema")

    def _ma(x: pd.Series, length: int, method: str) -> pd.Series:
        method = _normalize_method(method)
        if method == "sma":
            return x.rolling(length, min_periods=length).mean()
        if method == "ema":
            return x.ewm(span=length, adjust=False, min_periods=length).mean()
        if method == "smma":  # Wilder's / RMA
            alpha = 1.0 / float(length)
            return x.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
        if method == "lwma":
            w = np.arange(1, length + 1, dtype=float)
            w_sum = w.sum()
            return x.rolling(length, min_periods=length).apply(
                lambda a: np.dot(a, w) / w_sum, raw=True
            )
        # fallback
        return x.ewm(span=length, adjust=False, min_periods=length).mean()

    p = _applied_price(df, price)
    close = df["Close"].astype(float)

    ma1 = _ma(p, period, metod)
    ma2 = _ma(p, period, metod2).shift(1)

    threshold = float(porog) * float(tick_size)
    cond = (ma1.subtract(ma2).abs() >= threshold)

    # Build Signal: when cond True use ma2, else carry forward previous Signal
    signal_candidates = ma2.where(cond)
    signal = signal_candidates.ffill()

    # Initialize output columns
    up = pd.Series(np.nan, index=df.index, dtype=float)
    dn = pd.Series(np.nan, index=df.index, dtype=float)
    fl = pd.Series(np.nan, index=df.index, dtype=float)

    # Masks
    event = cond & signal.notna()
    non_event = (~cond) & signal.notna()

    up_mask = event & (close >= signal)
    dn_mask = event & (close <= signal)
    fl_mask_event = event & (close != signal)

    up.loc[up_mask] = signal.loc[up_mask]
    dn.loc[dn_mask] = signal.loc[dn_mask]
    fl.loc[fl_mask_event] = signal.loc[fl_mask_event]
    fl.loc[non_event] = signal.loc[non_event]

    out = pd.DataFrame(
        {
            "Signal": signal,
            "Fl": fl,
            "Up": up,
            "Dn": dn,
        },
        index=df.index,
    )
    return out