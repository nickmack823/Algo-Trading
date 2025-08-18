import numpy as np
import pandas as pd

def DecyclerOscillator(
    df: pd.DataFrame,
    hp_period: int = 125,
    k: float = 1.0,
    hp_period2: int = 100,
    k2: float = 1.2,
    price: str = "close",
) -> pd.DataFrame:
    """
    Simple Decycler Oscillator x 2.
    Returns a DataFrame with columns:
      - DEO: 100*k*HP( price - HP(price, hp_period), hp_period) / price
      - DEO2: 100*k2*HP( price - HP(price, hp_period2), hp_period2) / price
      - DEO2da, DEO2db: segmented DEO2 values for down-trend visualization (alternating segments)
    All outputs aligned to df.index; NaNs for warmup/undefined.

    Parameters:
      hp_period: int >= 2 (slow high-pass period)
      k: multiplier for DEO
      hp_period2: int >= 2 (fast high-pass period)
      k2: multiplier for DEO2
      price: one of
        close, open, high, low, median, typical, weighted, average, medianb, tbiased,
        haclose, haopen, hahigh, halow, hamedian, hatypical, haweighted, haaverage, hamedianb, hatbiased
    """
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    n = len(df)

    def _heiken_ashi(o, h, l, c):
        ha_close = (o + h + l + c) / 4.0
        ha_open = np.empty_like(ha_close)
        ha_open[:] = np.nan
        # Init
        ha_open[0] = (o[0] + c[0]) / 2.0
        # Recursive
        for i in range(1, len(o)):
            if np.isnan(ha_close[i - 1]) or np.isnan(ha_open[i - 1]):
                ha_open[i] = np.nan
            else:
                ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high = np.nanmax(np.vstack([h, ha_open, ha_close]), axis=0)
        ha_low = np.nanmin(np.vstack([l, ha_open, ha_close]), axis=0)
        return ha_open, ha_high, ha_low, ha_close

    def _select_price(kind: str) -> np.ndarray:
        k = kind.lower()
        if k == "close":
            return c
        if k == "open":
            return o
        if k == "high":
            return h
        if k == "low":
            return l
        if k == "median":
            return (h + l) / 2.0
        if k == "medianb":
            return (o + c) / 2.0
        if k == "typical":
            return (h + l + c) / 3.0
        if k == "weighted":
            return (h + l + 2.0 * c) / 4.0
        if k == "average":
            return (h + l + c + o) / 4.0
        if k == "tbiased":
            out = np.where(c > o, (h + c) / 2.0, (l + c) / 2.0)
            out[np.isnan(c) | np.isnan(o) | np.isnan(h) | np.isnan(l)] = np.nan
            return out
        # Heiken Ashi derived prices
        if k.startswith("ha"):
            ha_open, ha_high, ha_low, ha_close = _heiken_ashi(o, h, l, c)
            if k == "haclose":
                return ha_close
            if k == "haopen":
                return ha_open
            if k == "hahigh":
                return ha_high
            if k == "halow":
                return ha_low
            if k == "hamedian":
                return (ha_high + ha_low) / 2.0
            if k == "hamedianb":
                return (ha_open + ha_close) / 2.0
            if k == "hatypical":
                return (ha_high + ha_low + ha_close) / 3.0
            if k == "haweighted":
                return (ha_high + ha_low + 2.0 * ha_close) / 4.0
            if k == "haaverage":
                return (ha_high + ha_low + ha_close + ha_open) / 4.0
            if k == "hatbiased":
                out = np.where(ha_close > ha_open, (ha_high + ha_close) / 2.0, (ha_low + ha_close) / 2.0)
                out[np.isnan(ha_close) | np.isnan(ha_open) | np.isnan(ha_high) | np.isnan(ha_low)] = np.nan
                return out
        raise ValueError(f"Unsupported price type: {kind}")

    def _hp_2pole(x: np.ndarray, period: float) -> np.ndarray:
        # MQL iHp implementation
        out = np.empty_like(x)
        out[:] = np.nan
        if period is None or period <= 1 or len(x) == 0:
            # MQL returns 0 for all i when period<=1
            out = np.zeros_like(x, dtype=float)
            return out
        angle = 0.707 * 2.0 * np.pi / float(period)
        ca = np.cos(angle)
        sa = np.sin(angle)
        alpha = (ca + sa - 1.0) / ca
        a = (1.0 - alpha / 2.0) ** 2
        b = 2.0 * (1.0 - alpha)
        c2 = (1.0 - alpha) ** 2
        # Initialize first three values to 0 as per MQL (i<=2 -> 0)
        for i in range(len(x)):
            if i <= 2:
                out[i] = 0.0 if not np.isnan(x[i]) else np.nan
            else:
                p0, p1, p2 = x[i], x[i - 1], x[i - 2]
                y1, y2 = out[i - 1], out[i - 2]
                if np.isnan(p0) or np.isnan(p1) or np.isnan(p2) or np.isnan(y1) or np.isnan(y2):
                    out[i] = np.nan
                else:
                    out[i] = a * (p0 - 2.0 * p1 + p2) + b * y1 - c2 * y2
        return out

    px = _select_price(price)

    # DEO
    hp1_inner = _hp_2pole(px, hp_period)
    deo_num = _hp_2pole(px - hp1_inner, hp_period)
    deo = np.full(n, np.nan, dtype=float)
    np.divide(100.0 * k * deo_num, px, out=deo, where=(px != 0.0))

    # DEO2
    hp2_inner = _hp_2pole(px, hp_period2)
    deo2_num = _hp_2pole(px - hp2_inner, hp_period2)
    deo2 = np.full(n, np.nan, dtype=float)
    np.divide(100.0 * k2 * deo2_num, px, out=deo2, where=(px != 0.0))

    # Trend: sign(deo2 - deo) with tie -> carry forward previous
    diff = deo2 - deo
    trend = np.empty(n, dtype=float)
    trend[:] = np.nan
    for i in range(n):
        d = diff[i]
        if np.isnan(d):
            trend[i] = trend[i - 1] if i > 0 else np.nan
        else:
            if d > 0:
                trend[i] = 1.0
            elif d < 0:
                trend[i] = -1.0
            else:
                trend[i] = trend[i - 1] if i > 0 else np.nan

    # Segment down-trend (-1) into alternating groups -> two buffers
    idx = df.index
    trend_s = pd.Series(trend, index=idx)
    down_mask = trend_s.eq(-1.0)

    starts = down_mask & ~down_mask.shift(1, fill_value=False)
    group = starts.cumsum().where(down_mask, 0)  # 0 outside down segments
    # Alternate buffers by group parity
    grp_vals = group.to_numpy()
    deo2da = np.where((grp_vals > 0) & ((grp_vals % 2) == 1), deo2, np.nan)
    deo2db = np.where((grp_vals > 0) & ((grp_vals % 2) == 0), deo2, np.nan)

    out = pd.DataFrame(
        {
            "DEO": deo,
            "DEO2": deo2,
            "DEO2da": deo2da,
            "DEO2db": deo2db,
        },
        index=df.index,
    )
    return out