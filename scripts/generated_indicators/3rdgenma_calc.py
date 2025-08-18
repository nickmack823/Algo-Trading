import numpy as np
import pandas as pd

def ThirdGenMA(
    df: pd.DataFrame,
    ma_period: int = 220,
    sampling_period: int = 50,
    method: int = 1,
    applied_price: int = 5,
) -> pd.DataFrame:
    """
    3rd Generation Moving Average (Durschner).
    Returns:
      - MA3G: (alpha + 1) * MA1 - alpha * MA2
      - MA1:  first-pass MA of the applied price

    Params:
      - ma_period: main MA period (default 220)
      - sampling_period: sampling MA period for second pass (default 50)
      - method: 0=SMA, 1=EMA, 2=SMMA (Wilder/Smoothed), 3=LWMA (default 1)
      - applied_price: 0=Close,1=Open,2=High,3=Low,4=Median,5=Typical,6=Weighted (default 5)

    Output aligned to df.index and preserves length with NaNs for warmup.
    """
    if ma_period <= 0 or sampling_period <= 0:
        raise ValueError("ma_period and sampling_period must be positive integers.")
    if ma_period < 2 * sampling_period:
        raise ValueError("ma_period should be >= sampling_period * 2.")

    # Select applied price
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    if applied_price == 0:
        price = close
    elif applied_price == 1:
        price = open_
    elif applied_price == 2:
        price = high
    elif applied_price == 3:
        price = low
    elif applied_price == 4:
        price = (high + low) / 2.0
    elif applied_price == 5:
        price = (high + low + close) / 3.0
    elif applied_price == 6:
        price = (high + low + 2.0 * close) / 4.0
    else:
        price = close

    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).mean()

    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def _smma(s: pd.Series, n: int) -> pd.Series:
        # Wilder/Smoothed MA: seed with SMA(n), then recursive
        arr = s.to_numpy(dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        if len(arr) == 0:
            return pd.Series(out, index=s.index)

        seed = s.rolling(window=n, min_periods=n).mean().to_numpy()
        start = n - 1
        if start < len(arr):
            out[start] = seed[start]
            for i in range(start + 1, len(arr)):
                xi = arr[i]
                yi_1 = out[i - 1]
                if np.isnan(xi) or np.isnan(yi_1):
                    out[i] = np.nan
                else:
                    out[i] = (yi_1 * (n - 1) + xi) / n
        return pd.Series(out, index=s.index)

    def _lwma(s: pd.Series, n: int) -> pd.Series:
        weights = np.arange(1, n + 1, dtype=float)
        denom = weights.sum()
        # Using rolling.apply to handle NaNs within windows gracefully
        return s.rolling(window=n, min_periods=n).apply(
            lambda x: np.dot(x, weights) / denom, raw=True
        )

    def _ma(s: pd.Series, n: int, m: int) -> pd.Series:
        if m == 0:
            return _sma(s, n)
        elif m == 1:
            return _ema(s, n)
        elif m == 2:
            return _smma(s, n)
        elif m == 3:
            return _lwma(s, n)
        else:
            return _sma(s, n)

    # First pass MA
    ma1 = _ma(price, ma_period, method)

    # Second pass MA on the first MA (same method)
    ma2 = _ma(ma1, sampling_period, method)

    # Parameters
    lambda_ = float(ma_period) / float(sampling_period)
    denom = (ma_period - lambda_)
    # denom should be > 0 given earlier validation
    alpha = lambda_ * (ma_period - 1.0) / denom

    ma3g = (alpha + 1.0) * ma1 - alpha * ma2

    out = pd.DataFrame(
        {
            "MA3G": ma3g.to_numpy(dtype=float),
            "MA1": ma1.to_numpy(dtype=float),
        },
        index=df.index,
    )
    return out