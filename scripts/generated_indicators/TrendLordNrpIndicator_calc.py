import numpy as np
import pandas as pd

def TrendLordNrpIndicator(
    df: pd.DataFrame,
    length: int = 12,
    mode: str = "smma",
    price: str = "close",
    show_high_low: bool = False,
) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Computes the Trend Lord (non-repainting display logic) using two-stage MA:
    MA(length, mode) on price, then MA(sqrt(length), mode) on the first MA.
    Columns:
      - Buy: slot at Low/High (or MA) depending on up/down state
      - Sell: second-stage MA (trend line)
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=["Buy", "Sell"], dtype=float)

    n = max(int(length), 1)
    sqrt_n = max(int(np.sqrt(n)), 1)

    # Applied price
    p = price.lower()
    if p == "close":
        src = df["Close"]
    elif p == "open":
        src = df["Open"]
    elif p == "high":
        src = df["High"]
    elif p == "low":
        src = df["Low"]
    elif p == "median":
        src = (df["High"] + df["Low"]) / 2.0
    elif p == "typical":
        src = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif p == "weighted":
        src = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        raise ValueError("price must be one of: close, open, high, low, median, typical, weighted")

    m = mode.lower()

    def _ma(series: pd.Series, period: int, mth: str) -> pd.Series:
        if period <= 1:
            return series.astype(float)
        if mth in ("sma",):
            return series.rolling(window=period, min_periods=period).mean()
        elif mth in ("ema",):
            return series.ewm(span=period, adjust=False, min_periods=period).mean()
        elif mth in ("smma", "rma", "wilder"):
            alpha = 1.0 / period
            return series.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        elif mth in ("lwma", "wma"):
            weights = np.arange(1, period + 1, dtype=float)
            wsum = weights.sum()
            return series.rolling(window=period, min_periods=period).apply(
                lambda x: np.dot(x, weights) / wsum, raw=True
            )
        else:
            raise ValueError("mode must be one of: sma, ema, smma (rma/wilder), lwma")

    ma1 = _ma(src.astype(float), n, m)
    ma2 = _ma(ma1, sqrt_n, m)  # Array1 in MQL

    # Slots for Buy line
    if show_high_low:
        slot_ll = df["Low"].to_numpy(dtype=float)
        slot_hh = df["High"].to_numpy(dtype=float)
    else:
        slot_ll = ma1.to_numpy(dtype=float)
        slot_hh = ma1.to_numpy(dtype=float)

    # Direction based on current vs previous (MQL compares i to i+1)
    up = ma2 > ma2.shift(1)
    down = ma2 < ma2.shift(1)

    buy = np.full(len(df), np.nan, dtype=float)
    up_idx = up.fillna(False).to_numpy()
    dn_idx = down.fillna(False).to_numpy()
    buy[up_idx] = slot_ll[up_idx]
    buy[dn_idx] = slot_hh[dn_idx]

    sell = ma2.to_numpy(dtype=float)

    out = pd.DataFrame({"Buy": buy, "Sell": sell}, index=df.index)
    return out