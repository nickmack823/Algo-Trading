import numpy as np
import pandas as pd

def Supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Jason Robinson (2008) SuperTrend.
    Returns a DataFrame with columns:
      - 'Supertrend_Up': line when in uptrend (else NaN)
      - 'Supertrend_Down': line when in downtrend (else NaN)
      - 'Supertrend': merged primary line (Up or Down)
      - 'Trend': +1 for uptrend, -1 for downtrend (NaN during warmup)
    Vectorized where possible; uses Wilder ATR (ewm alpha=1/period). Preserves index and length with NaNs in warmup.
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=['Supertrend_Up', 'Supertrend_Down', 'Supertrend', 'Trend'], dtype='float64')

    h = df['High'].to_numpy(dtype='float64')
    l = df['Low'].to_numpy(dtype='float64')
    c = df['Close'].to_numpy(dtype='float64')

    # True Range
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    if len(c) > 1:
        prev_c[1:] = c[:-1]
    tr1 = h - l
    tr2 = np.abs(h - prev_c)
    tr3 = np.abs(l - prev_c)
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)
    tr_series = pd.Series(tr, index=df.index)

    # Wilder ATR via ewm(alpha=1/period), with warmup NaNs
    atr = tr_series.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean().to_numpy()

    median = (h + l) / 2.0
    up_base = median + multiplier * atr
    dn_base = median - multiplier * atr

    n = len(df)
    up = np.full(n, np.nan, dtype='float64')
    dn = np.full(n, np.nan, dtype='float64')
    trend = np.full(n, np.nan, dtype='float64')
    line_up = np.full(n, np.nan, dtype='float64')
    line_dn = np.full(n, np.nan, dtype='float64')

    started = False
    up_prev = np.nan
    dn_prev = np.nan
    trend_prev = 1  # default initial trend as in MQL

    for i in range(n):
        ub = up_base[i]
        db = dn_base[i]
        ci = c[i]

        # Require current bar values to proceed
        if np.isnan(ub) or np.isnan(db) or np.isnan(ci) or np.isnan(median[i]):
            continue

        if not started:
            # Seed with first available values
            trend_prev = 1
            up_prev = ub
            dn_prev = db
            trend[i] = 1
            dn[i] = db
            up[i] = ub
            line_up[i] = dn[i]
            started = True
            continue

        change = 0
        if ci > up_prev:
            curr_trend = 1
            if trend_prev == -1:
                change = 1
        elif ci < dn_prev:
            curr_trend = -1
            if trend_prev == 1:
                change = 1
        else:
            curr_trend = trend_prev
            change = 0

        flag = 1 if (curr_trend < 0 and trend_prev > 0) else 0
        flagh = 1 if (curr_trend > 0 and trend_prev < 0) else 0

        up_i = ub
        dn_i = db

        if curr_trend > 0 and dn_i < dn_prev:
            dn_i = dn_prev
        if curr_trend < 0 and up_i > up_prev:
            up_i = up_prev

        if flag == 1:
            up_i = ub
        if flagh == 1:
            dn_i = db

        if curr_trend == 1:
            line_up[i] = dn_i
            if change == 1 and i - 1 >= 0:
                line_up[i - 1] = line_dn[i - 1]
        else:
            line_dn[i] = up_i
            if change == 1 and i - 1 >= 0:
                line_dn[i - 1] = line_up[i - 1]

        trend[i] = curr_trend
        up[i] = up_i
        dn[i] = dn_i
        up_prev = up_i
        dn_prev = dn_i
        trend_prev = curr_trend

    supertrend = np.where(np.isnan(line_dn), line_up, line_dn)

    out = pd.DataFrame(
        {
            'Supertrend_Up': line_up,
            'Supertrend_Down': line_dn,
            'Supertrend': supertrend,
            'Trend': trend,
        },
        index=df.index,
    )
    return out