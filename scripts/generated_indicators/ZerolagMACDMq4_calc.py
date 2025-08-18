import numpy as np
import pandas as pd


def ZerolagMACDMq4(df: pd.DataFrame, fast: int = 12, slow: int = 24, signal: int = 9) -> pd.DataFrame:
    """Return ZeroLag MACD and Signal lines as columns aligned to df.index.
    Uses MT4-style EMA seeding (first value is SMA over the period); NaNs preserved during warmup."""
    close = pd.Series(df["Close"].astype(float).to_numpy(), index=df.index, dtype="float64")

    def _ema_mt4(s: pd.Series, period: int) -> pd.Series:
        if period is None or period <= 1:
            # EMA(1) equals the series itself
            out = s.astype("float64").copy()
            return out

        a = 2.0 / (period + 1.0)
        # Raw EMA (adjust=False => recursive form)
        ema_raw = s.ewm(alpha=a, adjust=False).mean()

        # Align to MT4 seeding: first EMA value equals SMA over the first 'period' valid observations
        roll = s.rolling(window=period, min_periods=period).mean()
        mask_valid = roll.notna().to_numpy()
        if not mask_valid.any():
            # Not enough data for initial SMA
            return pd.Series(np.nan, index=s.index, dtype="float64")

        pos0 = int(np.argmax(mask_valid))  # first index where SMA is available
        sma0 = float(roll.iloc[pos0])
        ema0 = float(ema_raw.iloc[pos0])
        delta = sma0 - ema0

        n = len(s)
        ema_adj = ema_raw.to_numpy(dtype="float64").copy()
        # NaN out everything before first valid EMA
        if pos0 > 0:
            ema_adj[:pos0] = np.nan
        # Exponential correction to enforce EMA[pos0] == SMA[pos0]
        decay = (1.0 - a)
        powers = np.power(decay, np.arange(n - pos0, dtype="float64"))
        ema_adj[pos0:] = ema_adj[pos0:] + delta * powers

        return pd.Series(ema_adj, index=s.index, dtype="float64")

    # First-level EMAs
    ema_fast = _ema_mt4(close, fast)
    ema_slow = _ema_mt4(close, slow)

    # Zero-lag EMAs for fast and slow: 2*EMA - EMA(EMA)
    ema_fast2 = _ema_mt4(ema_fast, fast)
    ema_slow2 = _ema_mt4(ema_slow, slow)
    zl_fast = 2.0 * ema_fast - ema_fast2
    zl_slow = 2.0 * ema_slow - ema_slow2

    zl_macd = zl_fast - zl_slow

    # Signal line: zero-lag EMA of MACD
    sig_ema = _ema_mt4(zl_macd, signal)
    sig_ema2 = _ema_mt4(sig_ema, signal)
    zl_signal = 2.0 * sig_ema - sig_ema2

    out = pd.DataFrame(
        {
            "ZL_MACD": zl_macd.to_numpy(dtype="float64"),
            "ZL_Signal": zl_signal.to_numpy(dtype="float64"),
        },
        index=df.index,
    )
    return out