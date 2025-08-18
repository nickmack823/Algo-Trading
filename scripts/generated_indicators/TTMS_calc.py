import numpy as np
import pandas as pd

def TTMS(
    df: pd.DataFrame,
    bb_length: int = 20,
    bb_deviation: float = 2.0,
    keltner_length: int = 20,
    keltner_smooth_length: int = 20,
    keltner_smooth_method: int = 0,  # 0-SMA, 1-EMA, 2-SMMA(Wilder), 3-LWMA
    keltner_deviation: float = 2.0,
) -> pd.DataFrame:
    """Return TTMS buffers as columns aligned to df.index."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n, min_periods=n).mean()

    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def _rma(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(alpha=1.0 / max(n, 1), adjust=False, min_periods=n).mean()

    def _lwma(s: pd.Series, n: int) -> pd.Series:
        if n <= 0:
            return pd.Series(np.nan, index=s.index)
        w = np.arange(1, n + 1, dtype=float)
        w_sum = w.sum()
        return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w_sum, raw=True)

    def _ma(s: pd.Series, n: int, method: int) -> pd.Series:
        if method == 0:
            return _sma(s, n)
        elif method == 1:
            return _ema(s, n)
        elif method == 2:
            return _rma(s, n)
        elif method == 3:
            return _lwma(s, n)
        else:
            return _sma(s, n)

    # Keltner center line MA
    ma_keltner = _ma(close, keltner_length, keltner_smooth_method)

    # ATR (Wilder)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = _rma(tr, keltner_smooth_length)

    # Keltner bands
    keltner_high = ma_keltner + atr * keltner_deviation
    keltner_low = ma_keltner - atr * keltner_deviation

    # Bollinger middle and std (population, MODE_SMA)
    bb_mid = _sma(close, bb_length)
    bb_std = close.rolling(bb_length, min_periods=bb_length).std(ddof=0)
    bb_top = bb_mid + bb_deviation * bb_std
    bb_bot = bb_mid - bb_deviation * bb_std

    # TTMS value
    denom = bb_top - bb_bot
    ttms_raw = (keltner_high - keltner_low) / denom - 1.0
    ttms = ttms_raw.where(denom != 0)

    # Split into Up/Dn histograms based on slope vs previous
    prev_ttms = ttms.shift(1)
    cond_up = ttms > prev_ttms
    ttms_up = ttms.where(cond_up, 0.0)
    ttms_dn = ttms.where(~cond_up, 0.0)
    valid_ttms = ttms.notna()
    ttms_up = ttms_up.where(valid_ttms)
    ttms_dn = ttms_dn.where(valid_ttms)

    # Alerts (squeeze on when BB is inside Keltner)
    valid_bands = keltner_high.notna() & keltner_low.notna() & bb_top.notna() & bb_bot.notna()
    squeeze_on = (bb_top < keltner_high) & (bb_bot > keltner_low) & valid_bands
    alert = pd.Series(np.where(squeeze_on, 1e-5, np.nan), index=df.index)
    noalert = pd.Series(np.where((~squeeze_on) & valid_bands, 1e-5, np.nan), index=df.index)

    out = pd.DataFrame(
        {
            "TTMS_Up": ttms_up,
            "TTMS_Dn": ttms_dn,
            "Alert": alert,
            "NoAlert": noalert,
        },
        index=df.index,
    )
    return out