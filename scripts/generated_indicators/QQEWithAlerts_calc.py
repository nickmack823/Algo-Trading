import numpy as np
import pandas as pd

def QQEWithAlerts(df: pd.DataFrame, rsi_period: int = 14, sf: int = 5) -> pd.DataFrame:
    """Return QQE lines as columns aligned to df.index: 
    - 'QQE_RSI_MA': EMA of RSI
    - 'QQE_TrendLevel': trailing level line
    Vectorized where possible; preserves NaNs for warmup."""
    close = pd.to_numeric(df['Close'], errors='coerce')

    # Wilder's RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # EMA of RSI with period sf
    rsi_ma = rsi.ewm(span=sf, adjust=False, min_periods=sf).mean()

    # ATR of RSI MA (absolute step)
    atr_rsi = (rsi_ma - rsi_ma.shift(1)).abs()

    # Wilders_Period used as EMA length to approximate Wilder's smoothing
    wilders_period = rsi_period * 2 - 1

    # Double-smoothed ATR of RSI
    ma_atr_rsi_1 = atr_rsi.ewm(span=wilders_period, adjust=False, min_periods=wilders_period).mean()
    ma_atr_rsi_2 = ma_atr_rsi_1.ewm(span=wilders_period, adjust=False, min_periods=wilders_period).mean()

    dar = ma_atr_rsi_2 * 4.236

    # Sequential computation of trailing level (TrLevelSlow)
    n = len(close)
    tr_level = np.full(n, np.nan, dtype=float)

    rsi_ma_np = rsi_ma.to_numpy(dtype=float)
    dar_np = dar.to_numpy(dtype=float)

    # Find the first index where we have valid values and a previous value
    valid = ~np.isnan(rsi_ma_np) & ~np.isnan(dar_np)
    if n >= 2:
        # Need both t and t-1 valid
        start_idx_candidates = np.where(valid & np.roll(valid, 1))[0]
        start_idx_candidates = start_idx_candidates[start_idx_candidates >= 1]
        if start_idx_candidates.size > 0:
            t0 = int(start_idx_candidates[0])
            tr = rsi_ma_np[t0 - 1]
            rsi1 = rsi_ma_np[t0 - 1]
            for t in range(t0, n):
                rsi0 = rsi_ma_np[t]
                d = dar_np[t]
                if np.isnan(rsi0) or np.isnan(d):
                    # keep NaN; carry forward rsi1 if possible
                    if not np.isnan(rsi0):
                        rsi1 = rsi0
                    continue
                dv = tr
                if rsi0 < tr:
                    tr = rsi0 + d
                    if rsi1 < dv and tr > dv:
                        tr = dv
                elif rsi0 > tr:
                    tr = rsi0 - d
                    if rsi1 > dv and tr < dv:
                        tr = dv
                # else: tr remains unchanged
                tr_level[t] = tr
                rsi1 = rsi0

    out = pd.DataFrame({
        'QQE_RSI_MA': rsi_ma,
        'QQE_TrendLevel': pd.Series(tr_level, index=df.index)
    }, index=df.index)

    return out