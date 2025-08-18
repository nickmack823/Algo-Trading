import numpy as np
import pandas as pd

def BandPassFilter(df: pd.DataFrame, period: int = 50, price: str = 'median', delta: float = 0.1) -> pd.DataFrame:
    """
    Ehlers Band-Pass Filter with slope/trend-segmented histogram outputs.

    Parameters
    - period: int, default 50
    - price: str, one of:
        'close','open','high','low','median','medianb','typical','weighted','average',
        'tbiased','tbiased2',
        'ha_close','ha_open','ha_high','ha_low','ha_median','ha_medianb','ha_typical','ha_weighted','ha_average','ha_tbiased','ha_tbiased2',
        'hab_close','hab_open','hab_high','hab_low','hab_median','hab_medianb','hab_typical','hab_weighted','hab_average','hab_tbiased','hab_tbiased2'
      default 'median'
    - delta: float, default 0.1

    Returns
    - DataFrame with columns:
        ['BP_StrongUp','BP_WeakUp','BP_StrongDown','BP_WeakDown','BP']
      aligned to df.index.
    """
    o = df['Open'].to_numpy(dtype=float)
    h = df['High'].to_numpy(dtype=float)
    l = df['Low'].to_numpy(dtype=float)
    c = df['Close'].to_numpy(dtype=float)
    n = len(df)

    period = int(max(1, period))
    delta = float(delta)

    prices = _compute_price(o, h, l, c, price)

    # Constants
    beta = np.cos(2.0 * np.pi / period)
    cos_term = np.cos(4.0 * np.pi * delta / period)
    # Avoid division by zero and numerical issues
    cos_term = np.where(np.isclose(cos_term, 0.0), np.finfo(float).tiny, cos_term)
    gamma = 1.0 / cos_term
    under = np.maximum(gamma * gamma - 1.0, 0.0)
    alpha = gamma - np.sqrt(under)

    # MQL4 code operates on series arrays (index 0 is most recent).
    # Convert to series-orientation by reversing, compute, then reverse back.
    p_s = prices[::-1]

    bp_s = np.full(n, np.nan, dtype=float)
    slope_s = np.zeros(n, dtype=float)
    trend_s = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        if i >= n - 2:
            bp_s[i] = p_s[i]
        else:
            x0 = p_s[i]
            x2 = p_s[i + 2]
            b1 = bp_s[i + 1]
            b2 = bp_s[i + 2]
            if not (np.isnan(x0) or np.isnan(x2) or np.isnan(b1) or np.isnan(b2) or np.isnan(alpha) or np.isnan(beta)):
                bp_s[i] = 0.5 * (1.0 - alpha) * (x0 - x2) + beta * (1.0 + alpha) * b1 - alpha * b2
            else:
                bp_s[i] = np.nan

        if i == n - 1:
            slope_s[i] = 0.0
            trend_s[i] = 0.0
        else:
            # slope
            if not (np.isnan(bp_s[i]) or np.isnan(bp_s[i + 1])):
                if bp_s[i] > bp_s[i + 1]:
                    slope_s[i] = 1.0
                elif bp_s[i] < bp_s[i + 1]:
                    slope_s[i] = -1.0
                else:
                    slope_s[i] = slope_s[i + 1]
            else:
                slope_s[i] = slope_s[i + 1]
            # trend
            if not np.isnan(bp_s[i]):
                if bp_s[i] > 0.0:
                    trend_s[i] = 1.0
                elif bp_s[i] < 0.0:
                    trend_s[i] = -1.0
                else:
                    trend_s[i] = trend_s[i + 1]
            else:
                trend_s[i] = trend_s[i + 1]

    # Reverse back to chronological order
    bp = bp_s[::-1]
    slope = slope_s[::-1]
    trend = trend_s[::-1]

    # Histograms
    strong_up = np.where((trend == 1.0) & (slope == 1.0), bp, np.nan)
    weak_up = np.where((trend == 1.0) & (slope == -1.0), bp, np.nan)
    strong_down = np.where((trend == -1.0) & (slope == -1.0), bp, np.nan)
    weak_down = np.where((trend == -1.0) & (slope == 1.0), bp, np.nan)

    out = pd.DataFrame({
        'BP_StrongUp': strong_up,
        'BP_WeakUp': weak_up,
        'BP_StrongDown': strong_down,
        'BP_WeakDown': weak_down,
        'BP': bp
    }, index=df.index)

    return out


def _compute_price(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, price: str) -> np.ndarray:
    p = (price or 'median').strip().lower().replace(' ', '').replace('-', '').replace('_', '')
    use_better = False
    is_ha = False

    base_map = {
        'close': c,
        'open': o,
        'high': h,
        'low': l,
        'median': (h + l) / 2.0,
        'medianb': (o + c) / 2.0,
        'typical': (h + l + c) / 3.0,
        'weighted': (h + l + 2.0 * c) / 4.0,
        'average': (h + l + c + o) / 4.0,
        'tbiased': np.where(c > o, (h + c) / 2.0, (l + c) / 2.0),
        'tbiased2': np.where(c > o, h, np.where(c < o, l, c)),
    }

    if p in base_map:
        return base_map[p].astype(float)

    if p.startswith('ha'):
        is_ha = True
        if p.startswith('hab'):
            use_better = True
            key = p[3:]  # after 'hab'
        else:
            key = p[2:]  # after 'ha'
        ha_o, ha_h, ha_l, ha_c = _heiken_ashi(o, h, l, c, use_better=use_better)
        if key == 'close':
            return ha_c
        if key == 'open':
            return ha_o
        if key == 'high':
            return ha_h
        if key == 'low':
            return ha_l
        if key == 'median':
            return (ha_h + ha_l) / 2.0
        if key == 'medianb':
            return (ha_o + ha_c) / 2.0
        if key == 'typical':
            return (ha_h + ha_l + ha_c) / 3.0
        if key == 'weighted':
            return (ha_h + ha_l + 2.0 * ha_c) / 4.0
        if key == 'average':
            return (ha_h + ha_l + ha_c + ha_o) / 4.0
        if key == 'tbiased':
            return np.where(ha_c > ha_o, (ha_h + ha_c) / 2.0, (ha_l + ha_c) / 2.0)
        if key == 'tbiased2':
            return np.where(ha_c > ha_o, ha_h, np.where(ha_c < ha_o, ha_l, ha_c))

    # Fallback to median if unknown price type
    return ((h + l) / 2.0).astype(float)


def _heiken_ashi(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, use_better: bool = False):
    n = len(o)
    ha_o = np.full(n, np.nan, dtype=float)
    ha_h = np.full(n, np.nan, dtype=float)
    ha_l = np.full(n, np.nan, dtype=float)
    ha_c = np.full(n, np.nan, dtype=float)

    for i in range(n):
        oi, hi, li, ci = o[i], h[i], l[i], c[i]
        if i == 0:
            prev_ha_o = (oi + ci) / 2.0 if not (np.isnan(oi) or np.isnan(ci)) else np.nan
        else:
            prev_ha_o = ha_o[i - 1]

        if np.isnan(oi) or np.isnan(hi) or np.isnan(li) or np.isnan(ci) or np.isnan(prev_ha_o):
            ha_o[i] = np.nan
            ha_c[i] = np.nan
            ha_h[i] = np.nan
            ha_l[i] = np.nan
            continue

        if use_better:
            if hi != li:
                ha_c_i = (oi + ci) / 2.0 + ((ci - oi) / (hi - li)) * abs((ci - oi) / 2.0)
            else:
                ha_c_i = (oi + ci) / 2.0
        else:
            ha_c_i = (oi + hi + li + ci) / 4.0

        ha_o_i = (prev_ha_o + (ha_c[i - 1] if i > 0 else (oi + ci) / 2.0)) / 2.0
        # The MQL version uses (prev_ha_open + prev_ha_close) / 2.0; our first bar uses (o+c)/2 as prev close proxy
        if i > 0 and not np.isnan(ha_c[i - 1]):
            ha_o_i = (prev_ha_o + ha_c[i - 1]) / 2.0
        else:
            ha_o_i = (prev_ha_o + (oi + ci) / 2.0) / 2.0

        hi_i = max(hi, ha_o_i, ha_c_i)
        li_i = min(li, ha_o_i, ha_c_i)

        ha_o[i] = ha_o_i
        ha_c[i] = ha_c_i
        ha_h[i] = hi_i
        ha_l[i] = li_i

    return ha_o, ha_h, ha_l, ha_c