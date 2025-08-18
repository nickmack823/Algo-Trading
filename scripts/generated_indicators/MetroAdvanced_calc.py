import numpy as np
import pandas as pd

def MetroAdvanced(
    df: pd.DataFrame,
    period_rsi: int = 14,
    rsi_type: str = "rsi",  # {"rsi","wilder","rsx","cutler"}
    price: str = "close",   # {"close","open","high","low","median","typical","weighted","median_body","average","trend_biased","volume"}
    step_size_fast: float = 5.0,
    step_size_slow: float = 15.0,
    over_sold: float = 10.0,
    over_bought: float = 90.0,
    minmax_period: int = 49
) -> pd.DataFrame:
    """
    Return ALL lines as columns aligned to df.index.
    Columns:
      RSI, StepRSI_fast, StepRSI_slow, Level_Up, Level_Mid, Level_Dn,
      TrendFast, TrendSlow, MinFast, MinSlow, MaxFast, MaxSlow, Trend
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(index=df.index, columns=[
            "RSI","StepRSI_fast","StepRSI_slow","Level_Up","Level_Mid","Level_Dn",
            "TrendFast","TrendSlow","MinFast","MinSlow","MaxFast","MaxSlow","Trend"
        ], dtype=float)

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float)

    price_key = str(price).lower()
    if price_key == "close":
        p = c
    elif price_key == "open":
        p = o
    elif price_key == "high":
        p = h
    elif price_key == "low":
        p = l
    elif price_key == "median":
        p = (h + l) / 2.0
    elif price_key == "typical":
        p = (h + l + c) / 3.0
    elif price_key == "weighted":
        p = (h + l + 2.0 * c) / 4.0
    elif price_key == "median_body":
        p = (o + c) / 2.0
    elif price_key == "average":
        p = (h + l + c + o) / 4.0
    elif price_key == "trend_biased":
        p = np.where(o > c, (h + c) / 2.0, (l + c) / 2.0)
    elif price_key == "volume":
        p = v
    else:
        p = c

    p = p.astype(float)
    # RSI computation (matching MQL4 logic)
    rsi_type_key = str(rsi_type).lower()
    rsi = np.full(n, np.nan, dtype=float)

    if rsi_type_key in ("rsi", "regular", "regular_rsi"):
        # Custom "regular" per source: EMA(alpha=1/period) of change and abs(change) with special warmup
        alpha = 1.0 / max(float(period_rsi), 1.0)
        change_ema = 0.0
        abs_change_ema = 0.0
        for t in range(n):
            if t == 0 or not np.isfinite(p[t]) or not np.isfinite(p[t-1]):
                delta = 0.0 if t == 0 else (p[t] - p[t-1])
            else:
                delta = p[t] - p[t-1]

            if t < period_rsi:
                # warmup as in MQL: average change from first and average abs change over available data
                kmax = min(period_rsi, t + 1)  # number of points including current
                k = max(kmax - 1, 0)
                if k == 0:
                    avg_change = 0.0
                    avg_abs = 0.0
                else:
                    # sum |p[t-j] - p[t-j-1]| for j=0..k-1
                    diffs = p[max(0, t - k + 1):t + 1] - p[max(0, t - k):t]
                    diffs = diffs[np.isfinite(diffs)]
                    if diffs.size == 0:
                        avg_abs = 0.0
                        avg_change = 0.0
                    else:
                        avg_abs = np.abs(diffs).sum() / k
                        # (p[t] - p[0]) / k
                        if np.isfinite(p[t]) and np.isfinite(p[0]):
                            avg_change = (p[t] - p[0]) / k
                        else:
                            avg_change = 0.0
                change_ema = avg_change
                abs_change_ema = avg_abs
            else:
                change_ema = change_ema + alpha * (delta - change_ema)
                abs_change_ema = abs_change_ema + alpha * (abs(delta) - abs_change_ema)

            if abs_change_ema != 0.0:
                rsi[t] = 50.0 * (change_ema / abs_change_ema + 1.0)
            else:
                rsi[t] = 50.0

    elif rsi_type_key in ("wilder", "wilders", "wil", "rsi_wil"):
        # Wilder's RSI variant per source using SMMA with period = 0.5*(period-1)
        delta = np.diff(p, prepend=p[0])
        pos_in = np.maximum(delta, 0.0) * 0.5 + 0.5 * np.abs(delta)
        neg_in = np.maximum(-delta, 0.0) * 0.5 + 0.5 * np.abs(delta)
        # Note: in source pos = 0.5*(|d|+d) -> max(d,0); neg = 0.5*(|d|-d) -> max(-d,0)
        pos_in = np.maximum(delta, 0.0)
        neg_in = np.maximum(-delta, 0.0)
        smma_period = 0.5 * (float(period_rsi) - 1.0)
        smma_period = max(smma_period, 1e-12)  # avoid division by zero

        pos_smma = np.zeros(n, dtype=float)
        neg_smma = np.zeros(n, dtype=float)
        for t in range(n):
            if t < smma_period:
                pos_smma[t] = pos_in[t] if np.isfinite(pos_in[t]) else 0.0
                neg_smma[t] = neg_in[t] if np.isfinite(neg_in[t]) else 0.0
            else:
                pos_prev = pos_smma[t-1]
                neg_prev = neg_smma[t-1]
                x_pos = pos_in[t] if np.isfinite(pos_in[t]) else 0.0
                x_neg = neg_in[t] if np.isfinite(neg_in[t]) else 0.0
                pos_smma[t] = pos_prev + (x_pos - pos_prev) / smma_period
                neg_smma[t] = neg_prev + (x_neg - neg_prev) / smma_period

            denom = pos_smma[t] + neg_smma[t]
            if denom != 0.0:
                rsi[t] = 100.0 * pos_smma[t] / denom
            else:
                rsi[t] = 50.0

    elif rsi_type_key in ("rsx", "rsx_rsi", "rsi_rsx"):
        # RSX per source (Ehlers-like IIR)
        Kg = 3.0 / (2.0 + float(period_rsi))
        Hg = 1.0 - Kg
        # states for mom (m1..m6) and |mom| (a1..a6)
        m1 = m2 = m3 = m4 = m5 = m6 = 0.0
        a1 = a2 = a3 = a4 = a5 = a6 = 0.0
        for t in range(n):
            if t == 0 or not np.isfinite(p[t]) or not np.isfinite(p[t-1]):
                mom = 0.0
            else:
                mom = p[t] - p[t-1]
            moa = abs(mom)

            if t < period_rsi:
                # reset states as in MQL for r<period
                m1 = m2 = m3 = m4 = m5 = m6 = 0.0
                a1 = a2 = a3 = a4 = a5 = a6 = 0.0
                rsi[t] = 50.0
                continue

            # cascade 1
            m1 = Kg * mom + Hg * m1
            m2 = Kg * m1 + Hg * m2
            mom_eff = 1.5 * m1 - 0.5 * m2
            a1 = Kg * moa + Hg * a1
            a2 = Kg * a1 + Hg * a2
            moa_eff = 1.5 * a1 - 0.5 * a2
            # cascade 2
            m3 = Kg * mom_eff + Hg * m3
            m4 = Kg * m3 + Hg * m4
            mom_eff = 1.5 * m3 - 0.5 * m4
            a3 = Kg * moa_eff + Hg * a3
            a4 = Kg * a3 + Hg * a4
            moa_eff = 1.5 * a3 - 0.5 * a4
            # cascade 3
            m5 = Kg * mom_eff + Hg * m5
            m6 = Kg * m5 + Hg * m6
            mom_eff = 1.5 * m5 - 0.5 * m6
            a5 = Kg * moa_eff + Hg * a5
            a6 = Kg * a5 + Hg * a6
            moa_eff = 1.5 * a5 - 0.5 * a6

            if moa_eff != 0.0:
                rsi[t] = np.clip((mom_eff / moa_eff + 1.0) * 50.0, 0.0, 100.0)
            else:
                rsi[t] = 50.0

    elif rsi_type_key in ("cutler", "cutlers", "cut", "rsi_cut"):
        delta = np.diff(p, prepend=p[0])
        pos = np.where(delta > 0.0, delta, 0.0)
        neg = np.where(delta < 0.0, -delta, 0.0)
        pos_roll = pd.Series(pos).rolling(period_rsi, min_periods=period_rsi).sum().to_numpy()
        neg_roll = pd.Series(neg).rolling(period_rsi, min_periods=period_rsi).sum().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(neg_roll > 0.0, pos_roll / neg_roll, np.nan)
        rsi = np.where(np.isfinite(rs), 100.0 - 100.0 / (1.0 + rs), np.nan)
        # per MQL return 50 when denominator is 0 or not enough data
        rsi = np.where(np.isfinite(rsi), rsi, 50.0)
    else:
        # default to custom "rsi" mode
        alpha = 1.0 / max(float(period_rsi), 1.0)
        change_ema = 0.0
        abs_change_ema = 0.0
        for t in range(n):
            if t == 0 or not np.isfinite(p[t]) or not np.isfinite(p[t-1]):
                delta = 0.0 if t == 0 else (p[t] - p[t-1])
            else:
                delta = p[t] - p[t-1]

            if t < period_rsi:
                kmax = min(period_rsi, t + 1)
                k = max(kmax - 1, 0)
                if k == 0:
                    avg_change = 0.0
                    avg_abs = 0.0
                else:
                    diffs = p[max(0, t - k + 1):t + 1] - p[max(0, t - k):t]
                    diffs = diffs[np.isfinite(diffs)]
                    if diffs.size == 0:
                        avg_abs = 0.0
                        avg_change = 0.0
                    else:
                        avg_abs = np.abs(diffs).sum() / k
                        if np.isfinite(p[t]) and np.isfinite(p[0]):
                            avg_change = (p[t] - p[0]) / k
                        else:
                            avg_change = 0.0
                change_ema = avg_change
                abs_change_ema = avg_abs
            else:
                change_ema = change_ema + alpha * (delta - change_ema)
                abs_change_ema = abs_change_ema + alpha * (abs(delta) - abs_change_ema)

            if abs_change_ema != 0.0:
                rsi[t] = 50.0 * (change_ema / abs_change_ema + 1.0)
            else:
                rsi[t] = 50.0

    # Step logic (fast/slow)
    maxf = rsi + 2.0 * step_size_fast
    minf = rsi - 2.0 * step_size_fast
    maxs = rsi + 2.0 * step_size_slow
    mins = rsi - 2.0 * step_size_slow

    trendf = np.zeros(n, dtype=float)
    trends = np.zeros(n, dtype=float)

    # We need recursive adjustments based on prior bar
    for t in range(n):
        if t == 0 or not np.isfinite(rsi[t]):
            # keep initial values as computed; trend states stay 0
            continue

        # Fast
        tf = trendf[t-1]
        maxf_prev = maxf[t-1]
        minf_prev = minf[t-1]
        if rsi[t] > maxf_prev:
            tf = 1.0
        if rsi[t] < minf_prev:
            tf = -1.0
        # adjust channel
        mf = minf[t]
        Mf = maxf[t]
        if tf > 0 and mf < minf_prev:
            mf = minf_prev
        if tf < 0 and Mf > maxf_prev:
            Mf = maxf_prev
        trendf[t] = tf
        minf[t] = mf
        maxf[t] = Mf

        # Slow
        ts = trends[t-1]
        maxs_prev = maxs[t-1]
        mins_prev = mins[t-1]
        if rsi[t] > maxs_prev:
            ts = 1.0
        if rsi[t] < mins_prev:
            ts = -1.0
        ms = mins[t]
        Ms = maxs[t]
        if ts > 0 and ms < mins_prev:
            ms = mins_prev
        if ts < 0 and Ms > maxs_prev:
            Ms = maxs_prev
        trends[t] = ts
        mins[t] = ms
        maxs[t] = Ms

    # Step lines
    step_fast_line = np.full(n, np.nan, dtype=float)
    step_slow_line = np.full(n, np.nan, dtype=float)
    pos_fast = trendf > 0
    neg_fast = trendf < 0
    step_fast_line[pos_fast] = minf[pos_fast] + step_size_fast
    step_fast_line[neg_fast] = maxf[neg_fast] - step_size_fast

    pos_slow = trends > 0
    neg_slow = trends < 0
    step_slow_line[pos_slow] = mins[pos_slow] + step_size_slow
    step_slow_line[neg_slow] = maxs[neg_slow] - step_size_slow

    # Levels based on rolling hi/lo of RSI over MinMaxPeriod
    rsi_s = pd.Series(rsi)
    hi = rsi_s.rolling(minmax_period, min_periods=1).max().to_numpy()
    lo = rsi_s.rolling(minmax_period, min_periods=1).min().to_numpy()
    rn = hi - lo
    level_up = lo + rn * (over_bought / 100.0)
    level_dn = lo + rn * (over_sold / 100.0)
    level_mid = 0.5 * (level_up + level_dn)

    # Final trend based on step lines
    trend = np.zeros(n, dtype=float)
    for t in range(n):
        s2 = step_fast_line[t]
        s3 = step_slow_line[t]
        if np.isfinite(s2) and np.isfinite(s3):
            if s2 > s3:
                trend[t] = 1.0
            elif s2 < s3:
                trend[t] = -1.0
            else:
                trend[t] = trend[t-1] if t > 0 else 0.0
        else:
            trend[t] = trend[t-1] if t > 0 else 0.0

    out = pd.DataFrame(
        {
            "RSI": rsi,
            "StepRSI_fast": step_fast_line,
            "StepRSI_slow": step_slow_line,
            "Level_Up": level_up,
            "Level_Mid": level_mid,
            "Level_Dn": level_dn,
            "TrendFast": trendf,
            "TrendSlow": trends,
            "MinFast": minf,
            "MinSlow": mins,
            "MaxFast": maxf,
            "MaxSlow": maxs,
            "Trend": trend,
        },
        index=df.index,
    )
    return out