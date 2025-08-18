import numpy as np
import pandas as pd

def DetrendedSyntheticPriceGoscillators(
    df: pd.DataFrame,
    dsp_period: int = 14,
    price_mode: str = "median",
    signal_period: int = 9,
    color_on: str = "outer",
) -> pd.DataFrame:
    """
    DSP oscillator (Mladen): returns all buffers:
      - DSP_LevelUp, DSP_LevelDown (outer EMAs gated by sign)
      - DSP (main line = EMA(alpha_m) - EMA(alpha_m/2))
      - DSP_Up_A, DSP_Up_B (alternating up-color segments)
      - DSP_Down_A, DSP_Down_B (alternating down-color segments)
      - DSP_State (int: -1,0,1)

    Parameters:
      - dsp_period: int, default 14
      - price_mode: one of
          close, open, high, low, median, typical, weighted, average, medianb,
          tbiased, tbiased2,
          haclose, haopen, hahigh, halow, hamedian, hatypical, haweighted,
          haaverage, hamedianb, hatbiased, hatbiased2
        (case-insensitive; 'pr_' prefixes accepted)
      - signal_period: int, default 9
      - color_on: one of outer, outer2, zero, slope (case-insensitive; 'chg_on' prefixes accepted)
    """
    idx = df.index
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "DSP_LevelUp",
                "DSP_LevelDown",
                "DSP",
                "DSP_Up_A",
                "DSP_Up_B",
                "DSP_Down_A",
                "DSP_Down_B",
                "DSP_State",
            ],
            index=idx,
        )

    O = df["Open"].astype(float).to_numpy(copy=False)
    H = df["High"].astype(float).to_numpy(copy=False)
    L = df["Low"].astype(float).to_numpy(copy=False)
    C = df["Close"].astype(float).to_numpy(copy=False)

    # Normalize params
    pm = price_mode.strip().lower()
    if pm.startswith("pr_"):
        pm = pm[3:]
    co = color_on.strip().lower()
    if co.startswith("chg_on"):
        co = co[6:] if color_on.lower().startswith("chg_on") else co

    # Heiken Ashi series (needs recursion for ha_open)
    ha_close = (O + H + L + C) / 4.0
    ha_open = np.full(n, np.nan)
    # initial ha_open as (open+close)/2 if available
    if n > 0:
        if np.isfinite(O[0]) and np.isfinite(C[0]):
            ha_open[0] = (O[0] + C[0]) / 2.0
    for t in range(1, n):
        if np.isfinite(ha_open[t - 1]) and np.isfinite(ha_close[t - 1]):
            ha_open[t] = 0.5 * (ha_open[t - 1] + ha_close[t - 1])
        else:
            # restart if direct values available, else remain NaN
            if np.isfinite(O[t]) and np.isfinite(C[t]):
                ha_open[t] = (O[t] + C[t]) / 2.0
    ha_high = np.maximum.reduce([H, ha_open, ha_close])
    ha_low = np.minimum.reduce([L, ha_open, ha_close])

    # Price selection
    def select_price(mode: str) -> np.ndarray:
        if mode == "close":
            return C
        if mode == "open":
            return O
        if mode == "high":
            return H
        if mode == "low":
            return L
        if mode == "median":
            return (H + L) / 2.0
        if mode == "medianb":
            return (O + C) / 2.0
        if mode == "typical":
            return (H + L + C) / 3.0
        if mode == "weighted":
            return (H + L + 2.0 * C) / 4.0
        if mode == "average":
            return (H + L + C + O) / 4.0
        if mode == "tbiased":
            return np.where(C > O, (H + C) / 2.0, (L + C) / 2.0)
        if mode == "tbiased2":
            return np.where(C > O, H, np.where(C < O, L, C))
        # Heiken Ashi derived
        if mode == "haclose":
            return ha_close
        if mode == "haopen":
            return ha_open
        if mode == "hahigh":
            return ha_high
        if mode == "halow":
            return ha_low
        if mode == "hamedian":
            return (ha_high + ha_low) / 2.0
        if mode == "hamedianb":
            return (ha_open + ha_close) / 2.0
        if mode == "hatypical":
            return (ha_high + ha_low + ha_close) / 3.0
        if mode == "haweighted":
            return (ha_high + ha_low + 2.0 * ha_close) / 4.0
        if mode == "haaverage":
            return (ha_high + ha_low + ha_close + ha_open) / 4.0
        if mode == "hatbiased":
            return np.where(ha_close > ha_open, (ha_high + ha_close) / 2.0, (ha_low + ha_close) / 2.0)
        if mode == "hatbiased2":
            return np.where(ha_close > ha_open, ha_high, np.where(ha_close < ha_open, ha_low, ha_close))
        raise ValueError(f"Unsupported price_mode: {price_mode}")

    px = select_price(pm)
    px_s = pd.Series(px, index=idx)

    # EMAs with specified alphas (ignore NaNs)
    alpha_m = 2.0 / (1.0 + float(dsp_period))
    ema_fast = px_s.ewm(alpha=alpha_m, adjust=False, ignore_na=True).mean()
    ema_slow = px_s.ewm(alpha=alpha_m / 2.0, adjust=False, ignore_na=True).mean()

    dsp = ema_fast - ema_slow  # main line
    v = dsp.to_numpy(copy=False)

    # Gated EMA levels
    alpha_s = 2.0 / (1.0 + float(signal_period))
    levelu = np.full(n, np.nan)
    leveld = np.full(n, np.nan)

    pu = 0.0
    pdn = 0.0
    for t in range(n):
        vt = v[t]
        if np.isfinite(vt):
            if vt > 0.0:
                pu = pu + alpha_s * (vt - pu)
            # else keep pu
            if vt < 0.0:
                pdn = pdn + alpha_s * (vt - pdn)
            # else keep pdn
        # if NaN, keep previous values
        levelu[t] = pu
        leveld[t] = pdn

    # State calculation
    state = np.zeros(n, dtype=int)
    if co == "outer":
        for t in range(n):
            vt = v[t]
            if np.isfinite(vt):
                if vt > levelu[t]:
                    state[t] = 1
                elif vt < leveld[t]:
                    state[t] = -1
                else:
                    state[t] = 0
            else:
                state[t] = 0
    elif co == "outer2":
        prev = 0
        for t in range(n):
            vt = v[t]
            cur = prev
            if np.isfinite(vt):
                if vt > levelu[t]:
                    cur = 1
                elif vt < leveld[t]:
                    cur = -1
            state[t] = cur
            prev = cur
    elif co == "zero":
        for t in range(n):
            vt = v[t]
            if np.isfinite(vt):
                state[t] = 1 if vt > 0 else (-1 if vt < 0 else 0)
            else:
                state[t] = 0
    elif co == "slope":
        prev = 0
        prev_v = np.nan
        for t in range(n):
            vt = v[t]
            cur = prev
            if np.isfinite(vt) and np.isfinite(prev_v):
                if vt > prev_v:
                    cur = 1
                elif vt < prev_v:
                    cur = -1
            state[t] = cur
            prev = cur
            prev_v = vt if np.isfinite(vt) else prev_v
    else:
        raise ValueError(f"Unsupported color_on: {color_on}")

    # Build alternating segment buffers for up and down states
    def alternating_segments(values: np.ndarray, mask: np.ndarray):
        # Identify starts of True runs
        prev = np.concatenate(([False], mask[:-1]))
        starts = mask & (~prev)
        seg_id = np.cumsum(starts.astype(np.int64))
        parity = seg_id % 2  # 1,0,1,0,...
        a_mask = mask & (parity == 0)
        b_mask = mask & (parity == 1)
        a = np.where(a_mask, values, np.nan)
        b = np.where(b_mask, values, np.nan)
        return a, b

    mask_up = state == 1
    mask_dn = state == -1

    up_a, up_b = alternating_segments(v, mask_up)
    dn_a, dn_b = alternating_segments(v, mask_dn)

    out = pd.DataFrame(
        {
            "DSP_LevelUp": pd.Series(levelu, index=idx, dtype=float),
            "DSP_LevelDown": pd.Series(leveld, index=idx, dtype=float),
            "DSP": dsp,
            "DSP_Up_A": pd.Series(up_a, index=idx, dtype=float),
            "DSP_Up_B": pd.Series(up_b, index=idx, dtype=float),
            "DSP_Down_A": pd.Series(dn_a, index=idx, dtype=float),
            "DSP_Down_B": pd.Series(dn_b, index=idx, dtype=float),
            "DSP_State": pd.Series(state, index=idx, dtype=int),
        },
        index=idx,
    )
    return out