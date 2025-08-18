import numpy as np
import pandas as pd

def VolatilityRatio(df: pd.DataFrame, period: int = 25, price: str = "Close") -> pd.DataFrame:
    """Return ALL lines as columns aligned to df.index:
       - VR: volatility ratio (std / SMA(std))
       - VR_below1_a: alternating segments of VR where VR < 1 (set A)
       - VR_below1_b: alternating segments of VR where VR < 1 (set B)
       Vectorized, handles NaNs, and preserves length with NaNs for warmup."""
    if period < 1:
        period = 1

    h, l, c = df["High"], df["Low"], df["Close"]
    p_lower = str(price).lower()
    if p_lower == "close":
        p = c
    elif p_lower == "open":
        p = df["Open"]
    elif p_lower == "high":
        p = h
    elif p_lower == "low":
        p = l
    elif p_lower in ("hl2", "median"):
        p = (h + l) / 2.0
    elif p_lower in ("hlc3", "typical"):
        p = (h + l + c) / 3.0
    elif p_lower in ("ohlc4",):
        p = (df["Open"] + h + l + c) / 4.0
    elif p_lower in ("weighted", "wclose", "wclose4"):
        p = (h + l + 2.0 * c) / 4.0
    else:
        # Fallback to close if unknown
        p = c

    # Population std over 'period' and its SMA over 'period'
    std = p.rolling(window=period, min_periods=period).std(ddof=0)
    ma_std = std.rolling(window=period, min_periods=period).mean()

    vr = std / ma_std
    vr = vr.where(ma_std.ne(0.0), 1.0)

    # Build alternating below-1 segments into two buffers
    mask = vr < 1.0
    run_start = mask & ~mask.shift(fill_value=False)
    run_id = run_start.cumsum()  # increases only at starts of True runs
    # For non-True positions, set run_id to 0 to simplify parity checks
    run_id_arr = np.where(mask.to_numpy(), run_id.to_numpy(), 0)

    is_a = mask & (run_id_arr % 2 == 1)
    is_b = mask & (run_id_arr % 2 == 0) & (run_id_arr != 0)

    vr_below_a = vr.where(is_a)
    vr_below_b = vr.where(is_b)

    out = pd.DataFrame(
        {
            "VR": vr.astype(float),
            "VR_below1_a": vr_below_a.astype(float),
            "VR_below1_b": vr_below_b.astype(float),
        },
        index=df.index,
    )
    return out