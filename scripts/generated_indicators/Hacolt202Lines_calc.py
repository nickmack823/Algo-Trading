import numpy as np
import pandas as pd

def Hacolt202Lines(df: pd.DataFrame, Length: int = 55, CandleSize: float = 1.1, LtLength: int = 60) -> pd.DataFrame:
    """Long-term Heikin-Ashi Candlestick Oscillator (HACO lt) by Sylvain Vervoort.
    Returns a DataFrame with columns ['HACOLT_Up','HACOLT_Dn','HACOLT'] aligned to df.index.
    Vectorized core with necessary stateful loops; handles NaNs and preserves warmup with NaNs."""
    o = df['Open'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)

    # Helpers
    def _tema(x: pd.Series, period: int) -> pd.Series:
        ema1 = x.ewm(span=period, adjust=False, min_periods=1).mean()
        ema2 = ema1.ewm(span=period, adjust=False, min_periods=1).mean()
        ema3 = ema2.ewm(span=period, adjust=False, min_periods=1).mean()
        return 3.0 * (ema1 - ema2) + ema3

    def _zero_lag_tema(x: pd.Series, period: int) -> pd.Series:
        t1 = _tema(x, period)
        t2 = _tema(t1, period)
        return 2.0 * t1 - t2

    n = len(df)
    if n == 0:
        return pd.DataFrame(index=df.index, columns=['HACOLT_Up', 'HACOLT_Dn', 'HACOLT'], dtype=float)

    # Heikin-Ashi components needed by the algo
    median = (h + l) / 2.0
    ha_close_prev = ((o + h + l + c) / 4.0).shift(1)
    # haOpen is an EMA of ha_close_prev with alpha=0.5: y_t = 0.5*x_t + 0.5*y_{t-1}
    ha_open = ha_close_prev.ewm(alpha=0.5, adjust=False, min_periods=1).mean()
    ha_c = (median + ha_open + pd.Series(np.maximum(ha_open.values, h.values), index=df.index)
            + pd.Series(np.minimum(ha_open.values, l.values), index=df.index)) / 4.0

    # Zero-lag TEMA on HA close and on median price
    temaa = _zero_lag_tema(ha_c, Length)
    temab = _zero_lag_tema(median, Length)
    delta = temab - temaa

    # Long-term EMA for ltSell condition
    ema_close_lt = c.ewm(span=LtLength, adjust=False, min_periods=1).mean()
    lt_sell = (c < ema_close_lt).values

    # Prepare numpy arrays for loop
    ha_c_np = ha_c.values
    ha_open_np = ha_open.values
    ha_c_prev = np.roll(ha_c_np, 1); ha_c_prev[0] = np.nan
    ha_open_prev = np.roll(ha_open_np, 1); ha_open_prev[0] = np.nan
    h_prev = np.roll(h.values, 1); h_prev[0] = np.nan
    l_prev = np.roll(l.values, 1); l_prev[0] = np.nan
    c_prev = np.roll(c.values, 1); c_prev[0] = np.nan

    o_np = o.values
    h_np = h.values
    l_np = l.values
    c_np = c.values
    median_np = median.values
    delta_np = delta.values

    # State arrays
    keeping1 = np.zeros(n, dtype=bool)
    keepall1 = np.zeros(n, dtype=bool)
    utr = np.zeros(n, dtype=bool)

    keeping2 = np.zeros(n, dtype=bool)
    keepall2 = np.zeros(n, dtype=bool)
    dtr = np.zeros(n, dtype=bool)

    upw = np.zeros(n, dtype=bool)
    dnw = np.zeros(n, dtype=bool)
    ltResult = np.zeros(n, dtype=bool)

    # Track last wave event (index and type)
    last_event_idx = -1
    last_event_is_up = False

    # Output values
    hacolt_val = np.full(n, np.nan, dtype=float)

    for t in range(n):
        # Up side conditions
        keep1_u = (
            (ha_c_np[t] > ha_open_np[t]) if np.all(np.isfinite([ha_c_np[t], ha_open_np[t]])) else False
        ) or (
            (ha_c_prev[t] >= ha_open_prev[t]) if np.all(np.isfinite([ha_c_prev[t], ha_open_prev[t]])) else False
        ) or (
            (c_np[t] >= ha_c_np[t]) if np.all(np.isfinite([c_np[t], ha_c_np[t]])) else False
        ) or (
            (h_np[t] > h_prev[t]) if np.all(np.isfinite([h_np[t], h_prev[t]])) else False
        ) or (
            (l_np[t] > l_prev[t]) if np.all(np.isfinite([l_np[t], l_prev[t]])) else False
        )
        keep2_u = (delta_np[t] >= 0) if np.isfinite(delta_np[t]) else False
        keep3_u = (
            (abs(c_np[t] - o_np[t]) < (h_np[t] - l_np[t]) * CandleSize) if np.all(np.isfinite([c_np[t], o_np[t], h_np[t], l_np[t]])) else False
        ) and (
            (h_np[t] >= l_prev[t]) if np.all(np.isfinite([h_np[t], l_prev[t]])) else False
        )

        keeping1[t] = keep1_u or keep2_u
        prev_keep1 = keeping1[t - 1] if t > 0 else False
        keepall1[t] = keeping1[t] or (prev_keep1 and ((c_np[t] >= o_np[t]) or ((c_np[t] >= c_prev[t]) if np.isfinite(c_prev[t]) else False)))
        prev_keepall1 = keepall1[t - 1] if t > 0 else False
        utr[t] = keepall1[t] or (prev_keepall1 and keep3_u)

        # Down side conditions
        keep1_d = (
            (ha_c_np[t] < ha_open_np[t]) if np.all(np.isfinite([ha_c_np[t], ha_open_np[t]])) else False
        ) or (
            (ha_c_prev[t] < ha_open_prev[t]) if np.all(np.isfinite([ha_c_prev[t], ha_open_prev[t]])) else False
        )
        keep2_d = (delta_np[t] < 0) if np.isfinite(delta_np[t]) else False
        keep3_d = (
            (abs(c_np[t] - o_np[t]) < (h_np[t] - l_np[t]) * CandleSize) if np.all(np.isfinite([c_np[t], o_np[t], h_np[t], l_np[t]])) else False
        ) and (
            (l_np[t] <= h_prev[t]) if np.all(np.isfinite([l_np[t], h_prev[t]])) else False
        )

        keeping2[t] = keep1_d or keep2_d
        prev_keep2 = keeping2[t - 1] if t > 0 else False
        keepall2[t] = keeping2[t] or (prev_keep2 and ((c_np[t] < o_np[t]) or ((c_np[t] < c_prev[t]) if np.isfinite(c_prev[t]) else False)))
        prev_keepall2 = keepall2[t - 1] if t > 0 else False
        dtr[t] = keepall2[t] or (prev_keepall2 and keep3_d)

        # Wave transitions
        prev_dtr = dtr[t - 1] if t > 0 else False
        prev_utr = utr[t - 1] if t > 0 else False
        upw[t] = (not dtr[t]) and prev_dtr and utr[t]
        dnw[t] = (not utr[t]) and prev_utr and dtr[t]

        # Update last event
        if upw[t]:
            last_event_idx = t
            last_event_is_up = True
        elif dnw[t]:
            last_event_idx = t
            last_event_is_up = False

        # Result logic
        result = False
        if upw[t]:
            result = True
        elif dnw[t]:
            result = False
        elif (last_event_idx >= 0) and ((t - last_event_idx) < 50) and last_event_is_up:
            result = True

        # Long-term result persistence
        ltResult[t] = ltResult[t - 1] if t > 0 else False
        if result:
            ltResult[t] = True
        elif (not result) and lt_sell[t]:
            ltResult[t] = False

        # Final HACOLT value: 1, 0, -1
        if result:
            hacolt_val[t] = 1.0
        elif ltResult[t]:
            hacolt_val[t] = 0.0
        else:
            hacolt_val[t] = -1.0

    # Warmup NaNs (triple EMA + LT EMA warmup)
    warmup = int(max(Length * 3, LtLength))
    if warmup > 0 and n > 0:
        warm_mask = np.zeros(n, dtype=bool)
        warm_mask[:min(warmup, n)] = True
    else:
        warm_mask = np.zeros(n, dtype=bool)

    hacolt = pd.Series(hacolt_val, index=df.index)
    hacolt[warm_mask] = np.nan
    hacolt_up = hacolt.where(hacolt > 0)
    hacolt_dn = hacolt.where(hacolt < 0)

    return pd.DataFrame({
        'HACOLT_Up': hacolt_up,
        'HACOLT_Dn': hacolt_dn,
        'HACOLT': hacolt
    }, index=df.index)