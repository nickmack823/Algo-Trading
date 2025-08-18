import numpy as np
import pandas as pd

def TheTurtleTradingChannel(
    df: pd.DataFrame,
    trade_period: int = 20,
    stop_period: int = 10,
    strict: bool = False,
) -> pd.DataFrame:
    """Return Turtle Trading Channel lines aligned to df.index.
    
    Parameters:
    - trade_period: Donchian channel period for trading signals (default 20)
    - stop_period: Donchian channel period for exit signals (default 10)
    - strict: If True, use intrabar HIGH/LOW breakouts; otherwise use CLOSE-based (except last bar)
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=[
            "UpperLine", "LowerLine", "LongsStopLine", "ShortsStopLine", "BullChange", "BearChange"
        ], dtype=float)

    H = df["High"].astype(float)
    L = df["Low"].astype(float)
    C = df["Close"].astype(float)
    n = len(df)

    # Donchian channels excluding current bar
    rhigh = H.rolling(trade_period, min_periods=trade_period).max().shift(1)
    rlow  = L.rolling(trade_period, min_periods=trade_period).min().shift(1)
    shigh = H.rolling(stop_period,  min_periods=stop_period).max().shift(1)
    slow  = L.rolling(stop_period,  min_periods=stop_period).min().shift(1)

    # Gating close-based signals on all bars except the last (i > 0 in MQL)
    not_last = np.ones(n, dtype=bool)
    if n > 0:
        not_last[-1] = False
    not_last_s = pd.Series(not_last, index=df.index)

    if strict:
        up_break = ((C > rhigh) & not_last_s) | (H > rhigh)
        dn_break = ((C < rlow) & not_last_s) | (L < rlow)
    else:
        up_break = (C > rhigh) & not_last_s
        dn_break = (C < rlow) & not_last_s

    # Prioritize up over down on simultaneous triggers (as in MQL's if/else if)
    sig = np.zeros(n, dtype=float)
    up_idx = up_break.fillna(False).to_numpy()
    dn_idx = dn_break.fillna(False).to_numpy()
    sig[up_idx] = 1.0
    sig[~up_idx & dn_idx] = -1.0

    # Regime/state: last non-zero signal carried forward
    state = pd.Series(np.where(sig != 0.0, sig, np.nan), index=df.index).ffill()

    prev_state = state.shift(1)

    # Lines
    upper_line = pd.Series(np.where(state.eq(1.0), rlow.to_numpy(), np.nan), index=df.index)
    lower_line = pd.Series(np.where(state.eq(-1.0), rhigh.to_numpy(), np.nan), index=df.index)
    longs_stop = pd.Series(np.where(state.eq(1.0), slow.to_numpy(), np.nan), index=df.index)
    shorts_stop = pd.Series(np.where(state.eq(-1.0), shigh.to_numpy(), np.nan), index=df.index)

    # Trend change markers
    bull_change = pd.Series(
        np.where((sig == 1.0) & (~prev_state.eq(1.0).fillna(True)), rlow.to_numpy(), np.nan),
        index=df.index
    )
    bear_change = pd.Series(
        np.where((sig == -1.0) & (~prev_state.eq(-1.0).fillna(True)), rhigh.to_numpy(), np.nan),
        index=df.index
    )

    out = pd.DataFrame(
        {
            "UpperLine": upper_line,
            "LowerLine": lower_line,
            "LongsStopLine": longs_stop,
            "ShortsStopLine": shorts_stop,
            "BullChange": bull_change,
            "BearChange": bear_change,
        },
        index=df.index,
    )
    return out