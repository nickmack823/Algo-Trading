import numpy as np
import pandas as pd

def GlitchIndexFixed(
    df: pd.DataFrame,
    MaPeriod: int = 30,
    MaMethod: str = "sma",
    Price: str = "median",
    level1: float = 1.0,
    level2: float = 1.0,
) -> pd.DataFrame:
    """
    Glitch Index Fixed (vectorized)
    Returns DataFrame with columns:
      ['gliUa','gliUb','gliDb','gliDa','gliNe','gli','state']
    """
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    n = len(df)

    # Helpers
    def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
        s = pd.Series(arr)
        return s.ewm(alpha=alpha, adjust=False).mean().to_numpy()

    def _smma(arr: np.ndarray, period: int) -> np.ndarray:
        # Wilder smoothing: y[t] = y[t-1] + (x[t] - y[t-1]) / period, seed with first non-nan
        x = arr.astype(float).copy()
        out = np.full_like(x, np.nan)
        if period <= 1:
            return x
        # find first finite
        idx = np.where(np.isfinite(x))[0]
        if idx.size == 0:
            return out
        start = idx[0]
        out[start] = x[start]
        for t in range(start + 1, len(x)):
            if np.isfinite(x[t]):
                prev = out[t - 1]
                if np.isfinite(prev):
                    out[t] = prev + (x[t] - prev) / period
                else:
                    out[t] = x[t]
            else:
                out[t] = out[t - 1]
        return out

    def _weighted_ma(arr: np.ndarray, weights: np.ndarray, use_mask: bool = True) -> np.ndarray:
        x = arr.astype(float)
        w = np.asarray(weights, dtype=float)
        x_filled = np.where(np.isfinite(x), x, 0.0)
        mask = np.isfinite(x).astype(float) if use_mask else np.ones_like(x, dtype=float)
        num = np.convolve(x_filled, w, mode="full")[: len(x)]
        den = np.convolve(mask, w, mode="full")[: len(x)]
        out = np.divide(num, den, out=np.full_like(x, np.nan), where=den != 0)
        return out

    def _lwma(arr: np.ndarray, period: int) -> np.ndarray:
        if period <= 1:
            return arr.astype(float)
        w = np.arange(period, 0, -1, dtype=float)  # current gets highest weight
        return _weighted_ma(arr, w, use_mask=True)

    def _slwma(arr: np.ndarray, period: int) -> np.ndarray:
        if period <= 1:
            return arr.astype(float)
        sqrt_p = int(np.floor(np.sqrt(period)))
        sqrt_p = max(sqrt_p, 1)
        first = _lwma(arr, period)
        second = _lwma(first, sqrt_p)
        return second

    def _sma(arr: np.ndarray, period: int) -> np.ndarray:
        s = pd.Series(arr)
        return s.rolling(window=int(max(1, period)), min_periods=1).mean().to_numpy()

    def _tema(arr: np.ndarray, period: int) -> np.ndarray:
        if period <= 1:
            return arr.astype(float)
        alpha = 2.0 / (1.0 + period)
        e1 = _ema(arr, alpha)
        e2 = _ema(e1, alpha)
        e3 = _ema(e2, alpha)
        return e3 + 3.0 * (e1 - e2)

    def _dsema(arr: np.ndarray, period: float) -> np.ndarray:
        if period <= 1:
            return arr.astype(float)
        alpha = 2.0 / (1.0 + np.sqrt(period))
        e1 = _ema(arr, alpha)
        e2 = _ema(e1, alpha)
        return e2

    def _lsma(arr: np.ndarray, period: int) -> np.ndarray:
        period = max(int(period), 1)
        lwma_avg = _lwma(arr, period)
        sma_avg = _sma(arr, period)
        out = 3.0 * lwma_avg - 2.0 * sma_avg
        # match MQL behavior: for r<period return price
        if period > 1:
            out[:period] = arr[:period]
        return out

    def _nlma(arr: np.ndarray, length: float) -> np.ndarray:
        x = arr.astype(float)
        if (length is None) or (length < 5) or (len(x) < 4):
            return x.copy()
        Cycle = 4.0
        Coeff = 3.0 * np.pi
        Phase = int(length - 1)
        L = int(length * 4) + Phase
        t = np.empty(L, dtype=float)
        for k in range(L):
            if k <= Phase - 1 and Phase > 1:
                t[k] = 1.0 * k / (Phase - 1)
            else:
                t[k] = 1.0 + (k - Phase + 1) * (2.0 * Cycle - 1.0) / (Cycle * length - 1.0)
        beta = np.cos(np.pi * t)
        g = np.where(t <= 0.5, 1.0, 1.0 / (Coeff * t + 1.0))
        w = g * beta
        w_sum = np.sum(w)
        # Convolution; divide by effective available weight (cap at w_sum for early edges)
        x_filled = np.where(np.isfinite(x), x, 0.0)
        mask = np.isfinite(x).astype(float)
        num = np.convolve(x_filled, w, mode="full")[: len(x)]
        eff = np.convolve(mask, w, mode="full")[: len(x)]
        # cap denominator at total theoretical weight (MQL divides by total for full windows; for partial/NaNs use available weight)
        den = np.minimum(eff, w_sum)
        out = np.divide(num, den, out=np.full_like(x, np.nan), where=den != 0)
        # for very early few bars, return price (MQL returns raw for r<3)
        out[:3] = x[:3]
        return out

    def _atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 50) -> np.ndarray:
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ])
        atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
        return atr

    # Price selection
    def _heiken_ashi(better: bool = False):
        ha_open = np.full(n, np.nan, dtype=float)
        ha_close = np.full(n, np.nan, dtype=float)
        ha_high = np.full(n, np.nan, dtype=float)
        ha_low = np.full(n, np.nan, dtype=float)

        for i in range(n):
            if better:
                if np.isfinite(h[i]) and np.isfinite(l[i]) and h[i] != l[i]:
                    hc = (o[i] + c[i]) / 2.0 + ((c[i] - o[i]) / (h[i] - l[i])) * np.abs((c[i] - o[i]) / 2.0)
                else:
                    hc = (o[i] + c[i]) / 2.0
            else:
                hc = (o[i] + h[i] + l[i] + c[i]) / 4.0
            if i == 0:
                ho = (o[i] + c[i]) / 2.0
            else:
                ho = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
            hh = np.nanmax([h[i], ho, hc])
            hl = np.nanmin([l[i], ho, hc])
            ha_open[i] = ho
            ha_close[i] = hc
            ha_high[i] = hh
            ha_low[i] = hl
        return ha_open, ha_high, ha_low, ha_close

    def _select_price(mode: str) -> np.ndarray:
        m = str(mode).strip().lower()
        if m in ("close", "pr_close"):
            return c.copy()
        if m in ("open", "pr_open"):
            return o.copy()
        if m in ("high", "pr_high"):
            return h.copy()
        if m in ("low", "pr_low"):
            return l.copy()
        if m in ("median", "pr_median"):
            return (h + l) / 2.0
        if m in ("medianb", "pr_medianb"):
            return (o + c) / 2.0
        if m in ("typical", "pr_typical"):
            return (h + l + c) / 3.0
        if m in ("weighted", "pr_weighted"):
            return (h + l + 2.0 * c) / 4.0
        if m in ("average", "pr_average"):
            return (h + l + c + o) / 4.0
        if m in ("tbiased", "pr_tbiased"):
            return np.where(c > o, (h + c) / 2.0, (l + c) / 2.0)
        if m in ("tbiased2", "pr_tbiased2"):
            out = c.copy()
            out[c > o] = h[c > o]
            out[c < o] = l[c < o]
            return out
        # Heiken Ashi
        if m.startswith("ha"):
            better = m.startswith("hab")
            ho, hh, hl, hc = _heiken_ashi(better=better)
            if m in ("haclose", "pr_haclose", "habclose", "pr_habclose"):
                return hc
            if m in ("haopen", "pr_haopen", "habopen", "pr_habopen"):
                return ho
            if m in ("hahigh", "pr_hahigh", "habhigh", "pr_habhigh"):
                return hh
            if m in ("halow", "pr_halow", "hablow", "pr_hablow"):
                return hl
            if m in ("hamedian", "pr_hamedian", "habmedian", "pr_habmedian"):
                return (hh + hl) / 2.0
            if m in ("hamedianb", "pr_hamedianb", "habmedianb", "pr_habmedianb"):
                return (ho + hc) / 2.0
            if m in ("hatypical", "pr_hatypical", "habtypical", "pr_habtypical"):
                return (hh + hl + hc) / 3.0
            if m in ("haweighted", "pr_haweighted", "habweighted", "pr_habweighted"):
                return (hh + hl + 2.0 * hc) / 4.0
            if m in ("haaverage", "pr_haaverage", "habaverage", "pr_habaverage"):
                return (hh + hl + hc + ho) / 4.0
            if m in ("hatbiased", "pr_hatbiased", "habtbiased", "pr_habtbiased"):
                return np.where(hc > ho, (hh + hc) / 2.0, (hl + hc) / 2.0)
            if m in ("hatbiased2", "pr_hatbiased2", "habtbiased2", "pr_habtbiased2"):
                out = hc.copy()
                out[hc > ho] = hh[hc > ho]
                out[hc < ho] = hl[hc < ho]
                return out
        # default
        return (h + l) / 2.0

    # Moving average selection
    def _ma(arr: np.ndarray, method: str, period: int) -> np.ndarray:
        m = str(method).strip().lower()
        if m in ("sma", "ma_sma"):
            return _sma(arr, int(np.ceil(period)))
        if m in ("ema", "ma_ema"):
            alpha = 2.0 / (1.0 + period)
            return _ema(arr, alpha)
        if m in ("smma", "ma_smma"):
            return _smma(arr, int(np.ceil(period)))
        if m in ("lwma", "ma_lwma"):
            return _lwma(arr, int(np.ceil(period)))
        if m in ("slwma", "ma_slwma"):
            return _slwma(arr, int(np.ceil(period)))
        if m in ("dsema", "ma_dsema"):
            return _dsema(arr, float(period))
        if m in ("tema", "ma_tema"):
            return _tema(arr, int(np.ceil(period)))
        if m in ("lsma", "ma_lsma"):
            return _lsma(arr, int(np.ceil(period)))
        if m in ("nlma", "ma_nlma"):
            return _nlma(arr, float(period))
        # default: return input
        return arr.astype(float)

    price_series = _select_price(Price)
    ma_series = _ma(price_series, MaMethod, int(MaPeriod))

    # ATR(50)
    atr = _atr_wilder(h, l, c, period=50)

    # Glitch Index
    gli = (c - ma_series) / atr
    gli = np.where(np.isfinite(gli), gli, np.nan)
    # Emulate MT4 warmup: compute only from index >= 49 (ensure ATR warmup)
    warm = 49
    valid_mask = np.arange(n) >= warm
    gli = np.where(valid_mask, gli, np.nan)

    # Buckets and state
    gliUa = np.where(gli > level2, gli, np.nan)
    gliUb = np.where((gli > level1) & (gli <= level2), gli, np.nan)
    gliDa = np.where(gli < -level2, gli, np.nan)
    gliDb = np.where((gli < -level1) & (gli >= -level2), gli, np.nan)
    gliNe = np.where((gli >= -level1) & (gli <= level1), gli, np.nan)

    state = np.full(n, np.nan)
    state = np.where(gli > level2, 2, state)
    state = np.where((gli > level1) & (gli <= level2), 1, state)
    state = np.where(gli < -level2, -2, state)
    state = np.where((gli < -level1) & (gli >= -level2), -1, state)
    state = np.where((gli >= -level1) & (gli <= level1), 0, state)

    out = pd.DataFrame(
        {
            "gliUa": gliUa,
            "gliUb": gliUb,
            "gliDb": gliDb,
            "gliDa": gliDa,
            "gliNe": gliNe,
            "gli": gli,
            "state": state,
        },
        index=df.index,
    )
    return out