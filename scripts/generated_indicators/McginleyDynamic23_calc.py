import numpy as np
import pandas as pd

def McginleyDynamic23(
    df: pd.DataFrame,
    period: int = 12,
    price: str = "close",
    constant: float = 5.0,
    method: str = "ema",
) -> pd.DataFrame:
    """
    McGinley Dynamic average (mladen version) with optional MA basis.
    Returns all plotted buffers as columns: mcg (main), mcg_down_a, mcg_down_b.

    Parameters:
    - period: int, default 12
    - price: {'close','open','high','low','median','typical','weighted'}, default 'close'
    - constant: float, default 5.0
    - method: {'sma','ema','smma','lwma','gen'}, default 'ema'
      'gen' is the average of SMA, EMA, SMMA, and LWMA (as in source).
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    price_key = str(price).lower()
    if price_key == "close":
        p = df["Close"].astype(float)
    elif price_key == "open":
        p = df["Open"].astype(float)
    elif price_key == "high":
        p = df["High"].astype(float)
    elif price_key == "low":
        p = df["Low"].astype(float)
    elif price_key == "median":
        p = (df["High"] + df["Low"]) / 2.0
    elif price_key == "typical":
        p = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif price_key == "weighted":
        p = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        raise ValueError("Unsupported price type")

    method_key = str(method).lower()

    # Helper MAs
    def ma_sma(x: pd.Series, n: int) -> pd.Series:
        return x.rolling(n, min_periods=n).mean()

    def ma_ema(x: pd.Series, n: int) -> pd.Series:
        return x.ewm(span=n, adjust=False, min_periods=1).mean()

    def ma_smma(x: pd.Series, n: int) -> pd.Series:
        # Wilder's smoothing approximated via EMA with alpha=1/n
        return x.ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean()

    def ma_lwma(x: pd.Series, n: int) -> pd.Series:
        if n == 1:
            return x.astype(float)
        w = np.arange(1, n + 1, dtype=float)
        w_sum = w.sum()
        return x.rolling(n, min_periods=n).apply(lambda a: np.dot(a, w) / w_sum, raw=True)

    if method_key == "sma":
        ma = ma_sma(p, period)
    elif method_key == "ema":
        ma = ma_ema(p, period)
    elif method_key == "smma":
        ma = ma_smma(p, period)
    elif method_key == "lwma":
        ma = ma_lwma(p, period)
    elif method_key == "gen":
        sma = ma_sma(p, period)
        ema = ma_ema(p, period)
        smma = ma_smma(p, period)
        lwma = ma_lwma(p, period)
        ma = (sma + ema + smma + lwma) / 4.0
    else:
        raise ValueError("Unsupported method")

    ma_shift = ma.shift(1)

    # McGinley Dynamic calculation
    # mcg[t] = ma[t-1] + (p[t] - ma[t-1]) / (constant * period * (p[t]/ma[t-1])^4)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p / ma_shift
        denom = constant * period * np.power(ratio, 4)
        mcg = ma_shift + (p - ma_shift) / denom

    # Mask invalid where MA unavailable or denom invalid
    invalid = (~np.isfinite(ma_shift)) | (~np.isfinite(denom)) | (ma_shift == 0.0)
    mcg = mcg.where(~invalid)

    # Down-slope segmentation into two alternating buffers
    neg = mcg < mcg.shift(1)
    starts = neg & ~neg.shift(1, fill_value=False)
    run_id = starts.cumsum()
    run_id = run_id.where(neg, 0)
    odd = (run_id % 2 == 1) & neg

    mcg_down_a = mcg.where(odd)
    mcg_down_b = mcg.where(neg & ~odd)

    out = pd.DataFrame(
        {
            "mcg": mcg.astype(float),
            "mcg_down_a": mcg_down_a.astype(float),
            "mcg_down_b": mcg_down_b.astype(float),
        },
        index=df.index,
    )
    return out