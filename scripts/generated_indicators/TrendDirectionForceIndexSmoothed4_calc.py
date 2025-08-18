import numpy as np
import pandas as pd

def TrendDirectionForceIndexSmoothed4(
    df: pd.DataFrame,
    trend_period: int = 20,
    trend_method: str = "ema",
    price: str = "close",
    trigger_up: float = 0.05,
    trigger_down: float = -0.05,
    smooth_length: float = 5.0,
    smooth_phase: float = 0.0,
    color_change_on_zero_cross: bool = False,
    point: float = 1.0,
) -> pd.DataFrame:
    """
    Return ALL lines as columns with clear names; aligned to df.index.
    Vectorized where possible; recursive parts are looped as needed. Handles NaNs.
    Supported trend_method: sma, ema, dsema, dema, tema, smma, lwma, pwma, vwma, hull, tma, sine, mcg, zlma, lead, ssm, smoo, linr, ilinr, ie2, nlma
    Supported price: close, open, high, low, median, typical, weighted, average, medianb, tbiased,
                     haclose, haopen, hahigh, halow, hamedian, hatypical, haweighted, haaverage, hamedianb, hatbiased
    """
    o = pd.Series(df["Open"].to_numpy(dtype=float), index=df.index)
    h = pd.Series(df["High"].to_numpy(dtype=float), index=df.index)
    l = pd.Series(df["Low"].to_numpy(dtype=float), index=df.index)
    c = pd.Series(df["Close"].to_numpy(dtype=float), index=df.index)
    v = pd.Series(df["Volume"].to_numpy(dtype=float), index=df.index)
    n = len(df)

    # Price selection
    def heikin_ashi(Open, High, Low, Close):
        ha_close = (Open + High + Low + Close) / 4.0
        ha_open = np.empty(n)
        ha_high = np.empty(n)
        ha_low = np.empty(n)
        ha_open[:] = np.nan
        ha_high[:] = np.nan
        ha_low[:] = np.nan
        for i in range(n):
            if i == 0 or not np.isfinite(ha_open[i - 1]):
                prev_ha_open = (Open.iloc[i] + Close.iloc[i]) / 2.0
                prev_ha_close = (Open.iloc[i] + High.iloc[i] + Low.iloc[i] + Close.iloc[i]) / 4.0
            else:
                prev_ha_open = ha_open[i - 1]
                prev_ha_close = ha_close.iloc[i - 1]
            hao = 0.5 * (prev_ha_open + prev_ha_close)
            hac = ha_close.iloc[i]
            hah = max(High.iloc[i], hao, hac)
            hal = min(Low.iloc[i], hao, hac)
            ha_open[i] = hao
            ha_high[i] = hah
            ha_low[i] = hal
        ha = pd.DataFrame(
            {
                "ha_open": pd.Series(ha_open, index=Open.index),
                "ha_high": pd.Series(ha_high, index=Open.index),
                "ha_low": pd.Series(ha_low, index=Open.index),
                "ha_close": ha_close,
            }
        )
        return ha

    def get_price(price_key: str) -> pd.Series:
        key = price_key.lower()
        if key.startswith("ha"):
            ha = heikin_ashi(o, h, l, c)
            if key == "haclose":
                p = ha["ha_close"]
            elif key == "haopen":
                p = ha["ha_open"]
            elif key == "hahigh":
                p = ha["ha_high"]
            elif key == "halow":
                p = ha["ha_low"]
            elif key == "hamedian":
                p = (ha["ha_high"] + ha["ha_low"]) / 2.0
            elif key == "hatypical":
                p = (ha["ha_high"] + ha["ha_low"] + ha["ha_close"]) / 3.0
            elif key == "haweighted":
                p = (ha["ha_high"] + ha["ha_low"] + 2.0 * ha["ha_close"]) / 4.0
            elif key == "haaverage":
                p = (ha["ha_high"] + ha["ha_low"] + ha["ha_close"] + ha["ha_open"]) / 4.0
            elif key == "hamedianb":
                p = (ha["ha_open"] + ha["ha_close"]) / 2.0
            elif key == "hatbiased":
                cond = ha["ha_close"] > ha["ha_open"]
                p = pd.Series(np.where(cond, (ha["ha_high"] + ha["ha_close"]) / 2.0, (ha["ha_low"] + ha["ha_close"]) / 2.0), index=ha.index)
            else:
                p = c.copy()
        else:
            if key == "close":
                p = c
            elif key == "open":
                p = o
            elif key == "high":
                p = h
            elif key == "low":
                p = l
            elif key == "median":
                p = (h + l) / 2.0
            elif key == "typical":
                p = (h + l + c) / 3.0
            elif key == "weighted":
                p = (h + l + 2.0 * c) / 4.0
            elif key == "average":
                p = (h + l + c + o) / 4.0
            elif key == "medianb":
                p = (o + c) / 2.0
            elif key == "tbiased":
                p = pd.Series(np.where(c > o, (h + c) / 2.0, (l + c) / 2.0), index=c.index)
            else:
                p = c
        return p.astype(float)

    x = get_price(price)

    # Rolling weighted helpers
    def _lwma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        w = np.arange(1.0, length + 1.0)
        denom = w.sum()
        return series.rolling(length, min_periods=length).apply(lambda a: float(np.dot(a, w) / denom), raw=True)

    def _pwma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        w = np.arange(1.0, length + 1.0) ** 2
        denom = w.sum()
        return series.rolling(length, min_periods=length).apply(lambda a: float(np.dot(a, w) / denom), raw=True)

    def _sma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        return series.rolling(length, min_periods=length).mean()

    def _ema(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        return series.ewm(span=length, adjust=False).mean()

    def _smma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr, dtype=float)
        out[:] = np.nan
        alpha = 1.0 / float(length)
        prev = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if not np.isfinite(prev):
                prev = xi
                out[i] = prev
            else:
                prev = prev + alpha * (xi - prev)
                out[i] = prev
        return pd.Series(out, index=series.index)

    def _vwma(series: pd.Series, vol: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        num = (series * vol).rolling(length, min_periods=length).sum()
        den = vol.rolling(length, min_periods=length).sum()
        return num / den

    def _tema(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        e1 = _ema(series, length)
        e2 = _ema(e1, length)
        e3 = _ema(e2, length)
        return 3.0 * (e1 - e2) + e3

    def _dema(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        e1 = _ema(series, length)
        e2 = _ema(e1, length)
        return 2.0 * e1 - e2

    def _dsema(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        L = max(length, 1.0)
        alpha = 2.0 / (1.0 + np.sqrt(L))
        arr = series.to_numpy(copy=True)
        e1 = np.empty_like(arr)
        e2 = np.empty_like(arr)
        e1[:] = np.nan
        e2[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                e1[i] = np.nan
                e2[i] = np.nan
                continue
            if i == 0 or not np.isfinite(e1[i - 1]):
                e1[i] = xi
                e2[i] = xi
            else:
                e1[i] = e1[i - 1] + alpha * (xi - e1[i - 1])
                e2[i] = e2[i - 1] + alpha * (e1[i] - e2[i - 1])
        return pd.Series(e2, index=series.index)

    def _tma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        n1 = int(np.ceil((length + 1) / 2.0))
        n2 = int(np.floor((length + 1) / 2.0))
        return _sma(_sma(series, n1), n2)

    def _hull(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        n1 = int(np.floor(length / 2.0))
        n2 = int(length)
        n3 = int(max(np.floor(np.sqrt(length)), 1))
        w1 = _lwma(series, n1)
        w2 = _lwma(series, n2)
        diff = 2.0 * w1 - w2
        return _lwma(diff, n3)

    def _sinewma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        k = np.arange(1.0, length + 1.0)
        w = np.sin(np.pi * k / (length + 1.0))
        denom = w.sum()
        return series.rolling(length, min_periods=length).apply(lambda a: float(np.dot(a, w) / denom), raw=True)

    def _mcg(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i == 0 or not np.isfinite(out[i - 1]) or out[i - 1] == 0:
                out[i] = xi
            else:
                denom = length * (xi / out[i - 1]) ** 4 / 2.0
                out[i] = out[i - 1] + (xi - out[i - 1]) / denom
        return pd.Series(out, index=series.index)

    def _zlma(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        L = int((length - 1.0) // 2)
        alpha = 2.0 / (1.0 + length)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i <= L:
                out[i] = xi
            else:
                if int(length) % 2 == 0:
                    median = (arr[i - L] + arr[i - L - 1]) / 2.0
                else:
                    median = arr[i - L]
                prev = out[i - 1] if np.isfinite(out[i - 1]) else xi
                out[i] = prev + alpha * (2.0 * xi - median - prev)
        return pd.Series(out, index=series.index)

    def _leader(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        alpha = 2.0 / (length + 1.0)
        arr = series.to_numpy(copy=True)
        e1 = np.empty_like(arr)
        e2 = np.empty_like(arr)
        e1[:] = np.nan
        e2[:] = np.nan
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i == 0 or not np.isfinite(e1[i - 1]):
                e1[i] = xi
                e2[i] = xi
            else:
                e1[i] = e1[i - 1] + alpha * (xi - e1[i - 1])
                e2[i] = e2[i - 1] + alpha * (xi - e1[i] - e2[i - 1])
            out[i] = e1[i] + e2[i]
        return pd.Series(out, index=series.index)

    def _ssm(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        a1 = np.exp(-1.414 * np.pi / length)
        b1 = 2.0 * a1 * np.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1.0 - c2 - c3
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i < 2 or not np.isfinite(out[i - 1]) or not np.isfinite(out[i - 2]):
                out[i] = xi
            else:
                out[i] = c1 * (xi + arr[i - 1]) / 2.0 + c2 * out[i - 1] + c3 * out[i - 2]
        return pd.Series(out, index=series.index)

    def _smooth_simple(series: pd.Series, length: int) -> pd.Series:
        # iSmooth(price,int length,...) simple variant used by "smoo"
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        s0 = np.empty_like(arr); s1 = np.empty_like(arr); s2 = np.empty_like(arr)
        s3 = np.empty_like(arr); s4 = np.empty_like(arr)
        s0[:] = np.nan; s1[:] = np.nan; s2[:] = np.nan; s3[:] = np.nan; s4[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                s4[i] = np.nan
                continue
            if i <= 2:
                s0[i] = xi; s2[i] = xi; s4[i] = xi
                s1[i] = 0.0; s3[i] = 0.0
            else:
                beta = 0.45 * (length - 1.0) / (0.45 * (length - 1.0) + 2.0)
                alpha = beta
                s0[i] = xi + alpha * (s0[i - 1] - xi)
                s1[i] = (xi - s0[i]) * (1 - alpha) + alpha * s1[i - 1]
                s2[i] = s0[i] + s1[i]
                s3[i] = (s2[i] - s4[i - 1]) * (1 - alpha) ** 2 + (alpha ** 2) * s3[i - 1]
                s4[i] = s3[i] + s4[i - 1]
        return pd.Series(s4, index=series.index)

    def _linr(series: pd.Series, length: int) -> pd.Series:
        # 3*LWMA - 2*SMA approximation per MQL implementation
        L = _lwma(series, length)
        S = _sma(series, length)
        return 3.0 * L - 2.0 * S

    def _ilinr(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr); out[:] = np.nan
        sma = np.empty_like(arr); sma[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan; continue
            if i + 1 >= length:
                window = arr[i - length + 1 : i + 1]
                idx = np.arange(length, dtype=float)
                sumx = idx.sum()
                sumxx = (idx ** 2).sum()
                sumy = np.nansum(window)
                sum1 = np.nansum(window * idx)
                slope = 0.0
                den = sumx * sumx - length * sumxx
                if den != 0:
                    slope = (sum1 * length - sumx * sumy) / den
                sma[i] = np.nanmean(window)
                out[i] = sma[i] + slope
            else:
                out[i] = xi
        return pd.Series(out, index=series.index)

    def _ie2(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr); out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan; continue
            if i + 1 >= length:
                window = arr[i - length + 1 : i + 1]
                idx = np.arange(length, dtype=float)
                sumx = idx.sum()
                sumxx = (idx ** 2).sum()
                sumxy = np.nansum(window * idx)
                sumy = np.nansum(window)
                den = sumx * sumx - length * sumxx
                tslope = (length * sumxy - sumx * sumy) / den if den != 0 else 0.0
                average = sumy / length
                out[i] = ((average + tslope) + (sumy + tslope * sumx) / length) / 2.0
            else:
                out[i] = xi
        return pd.Series(out, index=series.index)

    def _nlma(series: pd.Series, length: float) -> pd.Series:
        L = float(length)
        if L < 3 or n == 0:
            return series.astype(float)
        Cycle = 4.0
        Coeff = 3.0 * np.pi
        Phase = int(L - 1)
        total_len = int(L * 4 + Phase)
        # Precompute alphas/weights
        alphas = np.empty(total_len)
        for k in range(total_len):
            if k <= Phase - 1:
                t = 1.0 * k / max(Phase - 1, 1)
            else:
                t = 1.0 + (k - Phase + 1) * (2.0 * Cycle - 1.0) / (Cycle * L - 1.0)
            beta = np.cos(np.pi * t)
            g = 1.0 if t <= 0.5 else 1.0 / (Coeff * t + 1.0)
            alphas[k] = g * beta
        wsum = alphas.sum()
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr); out[:] = np.nan
        for i in range(n):
            if i == 0 or not np.isfinite(arr[i]):
                out[i] = arr[i] if np.isfinite(arr[i]) else np.nan
                continue
            acc = 0.0
            kk = 0
            j = i
            while kk < total_len and j >= 0:
                val = arr[j]
                if np.isfinite(val):
                    acc += alphas[kk] * val
                kk += 1
                j -= 1
            if wsum != 0:
                out[i] = acc / wsum
            else:
                out[i] = 0.0
        return pd.Series(out, index=series.index)

    ma_key = trend_method.lower()
    if ma_key == "sma":
        mma = _sma(x, int(trend_period))
    elif ma_key == "ema":
        mma = _ema(x, float(trend_period))
    elif ma_key == "dsema":
        mma = _dsema(x, float(trend_period))
    elif ma_key == "dema":
        mma = _dema(x, float(trend_period))
    elif ma_key == "tema":
        mma = _tema(x, float(trend_period))
    elif ma_key == "smma":
        mma = _smma(x, int(trend_period))
    elif ma_key == "lwma":
        mma = _lwma(x, int(trend_period))
    elif ma_key == "pwma":
        mma = _pwma(x, int(trend_period))
    elif ma_key == "vwma":
        mma = _vwma(x, v, int(trend_period))
    elif ma_key == "hull":
        mma = _hull(x, int(trend_period))
    elif ma_key == "tma":
        mma = _tma(x, int(trend_period))
    elif ma_key == "sine":
        mma = _sinewma(x, int(trend_period))
    elif ma_key == "mcg":
        mma = _mcg(x, float(trend_period))
    elif ma_key == "zlma":
        mma = _zlma(x, float(trend_period))
    elif ma_key == "lead":
        mma = _leader(x, float(trend_period))
    elif ma_key == "ssm":
        mma = _ssm(x, float(trend_period))
    elif ma_key == "smoo":
        mma = _smooth_simple(x, int(trend_period))
    elif ma_key == "linr":
        mma = _linr(x, int(trend_period))
    elif ma_key == "ilinr":
        mma = _ilinr(x, int(trend_period))
    elif ma_key == "ie2":
        mma = _ie2(x, int(trend_period))
    elif ma_key == "nlma":
        mma = _nlma(x, float(trend_period))
    else:
        raise ValueError(f"Unsupported trend_method '{trend_method}'")

    # SMMA of MMA with alpha = 2/(trend_period+1) (EMA-like on MMA)
    alpha = 2.0 / (float(trend_period) + 1.0) if trend_period > 0 else 1.0
    smma2 = mma.ewm(alpha=alpha, adjust=False).mean()

    impet_mma = mma.diff()
    impet_smma = smma2.diff()
    divma = (mma - smma2).abs() / float(point)
    averimpet = (impet_mma + impet_smma) / (2.0 * float(point))
    tdf_raw = divma * (averimpet ** 3)

    # Normalize by rolling max(abs) over trend_period*3
    norm_len = max(int(trend_period * 3), 1)
    absmax = tdf_raw.abs().rolling(norm_len, min_periods=1).max()
    normalized = pd.Series(np.where(absmax.to_numpy() > 0, (tdf_raw / absmax).to_numpy(), 0.0), index=tdf_raw.index)

    # Final smoothing with variable-phase iSmooth (length, phase)
    def _iSmooth_variable(series: pd.Series, length: float, phase: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        # wrk columns: [0..4] + [bsmax, bsmin, volty, vsum, avolty] = 10 total
        wrk = np.zeros((n, 10), dtype=float)
        out = np.empty(n, dtype=float); out[:] = np.nan
        for r in range(n):
            price_r = arr[r]
            if not np.isfinite(price_r):
                out[r] = np.nan
                if r > 0:
                    wrk[r] = wrk[r - 1]
                continue
            if r == 0:
                wrk[r, 0:7] = price_r
                wrk[r, 7:] = 0.0
                out[r] = price_r
                continue
            len1 = max(np.log(np.sqrt(0.5 * (length - 1))) / np.log(2.0) + 2.0, 0.0)
            pow1 = max(len1 - 2.0, 0.5)
            del1 = price_r - wrk[r - 1, 5]  # bsmax
            del2 = price_r - wrk[r - 1, 6]  # bsmin
            div = 1.0 / (10.0 + 10.0 * (min(max(length - 10.0, 0.0), 100.0)) / 100.0)
            forBar = min(r, 10)
            volty = max(abs(del1), abs(del2))
            wrk[r, 7] = wrk[r - 1, 7] + (volty - wrk[r - forBar, 7]) * div  # vsum
            av_prev = wrk[r - 1, 9]
            wrk[r, 9] = av_prev + (2.0 / (max(4.0 * length, 30.0) + 1.0)) * (wrk[r, 7] - av_prev)
            dVolty = 0.0
            if wrk[r, 9] > 0:
                dVolty = volty / wrk[r, 9]
            dVolty = min(max(dVolty, 1.0), len1 ** (1.0 / pow1))
            pow2 = dVolty ** pow1
            len2 = np.sqrt(0.5 * (length - 1.0)) * len1
            Kv = (len2 / (len2 + 1.0)) ** np.sqrt(pow2)
            wrk[r, 5] = price_r if del1 > 0 else price_r - Kv * del1  # bsmax
            wrk[r, 6] = price_r if del2 < 0 else price_r - Kv * del2  # bsmin
            R = np.clip(phase, -100.0, 100.0) / 100.0 + 1.5
            beta = 0.45 * (length - 1.0) / (0.45 * (length - 1.0) + 2.0)
            alpha_v = beta ** pow2
            wrk[r, 0] = price_r + alpha_v * (wrk[r - 1, 0] - price_r)
            wrk[r, 1] = (price_r - wrk[r, 0]) * (1.0 - beta) + beta * wrk[r - 1, 1]
            wrk[r, 2] = wrk[r, 0] + R * wrk[r, 1]
            wrk[r, 3] = (wrk[r, 2] - wrk[r - 1, 4]) * (1.0 - alpha_v) ** 2 + (alpha_v ** 2) * wrk[r - 1, 3]
            wrk[r, 4] = wrk[r - 1, 4] + wrk[r, 3]
            out[r] = wrk[r, 4]
        return pd.Series(out, index=series.index)

    tdf_smoothed = _iSmooth_variable(normalized, float(smooth_length), float(smooth_phase))

    # Trend states
    trend = np.zeros(n, dtype=float)
    tb = tdf_smoothed.to_numpy(copy=True)
    for i in range(n):
        prev = trend[i - 1] if i > 0 else 0.0
        val = tb[i]
        if not np.isfinite(val):
            trend[i] = prev
            continue
        if color_change_on_zero_cross:
            if val > 0:
                trend[i] = 1.0
            elif val < 0:
                trend[i] = -1.0
            else:
                trend[i] = prev
        else:
            if val > trigger_up:
                trend[i] = 1.0
            elif val < trigger_down:
                trend[i] = -1.0
            else:
                trend[i] = 0.0

    # Segmented up/down plot arrays (emulate PlotPoint backward)
    def _plot_segments(trend_arr: np.ndarray, src: np.ndarray):
        up_a = np.full(n, np.nan)
        up_b = np.full(n, np.nan)
        dn_a = np.full(n, np.nan)
        dn_b = np.full(n, np.nan)

        def plot_point(i, first, second, source):
            if i >= n - 2:
                return
            if np.isnan(first[i + 1]):
                if np.isnan(first[i + 2]):
                    first[i] = source[i]
                    first[i + 1] = source[i + 1]
                    second[i] = np.nan
                else:
                    second[i] = source[i]
                    second[i + 1] = source[i + 1]
                    first[i] = np.nan
            else:
                first[i] = source[i]
                second[i] = np.nan

        for i in range(n - 1, -1, -1):
            if not np.isfinite(src[i]):
                continue
            if trend_arr[i] == 1.0:
                plot_point(i, up_a, up_b, src)
            elif trend_arr[i] == -1.0:
                plot_point(i, dn_a, dn_b, src)
        return up_a, up_b, dn_a, dn_b

    up_a, up_b, dn_a, dn_b = _plot_segments(trend, tb)

    out = pd.DataFrame(
        {
            "TDF": tdf_smoothed,
            "TriggerUp": pd.Series(np.full(n, trigger_up), index=df.index),
            "TriggerDown": pd.Series(np.full(n, trigger_down), index=df.index),
            "UpA": pd.Series(up_a, index=df.index),
            "UpB": pd.Series(up_b, index=df.index),
            "DownA": pd.Series(dn_a, index=df.index),
            "DownB": pd.Series(dn_b, index=df.index),
            "Trend": pd.Series(trend, index=df.index),
        },
        index=df.index,
    )
    return out