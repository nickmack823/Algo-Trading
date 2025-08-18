import numpy as np
import pandas as pd

def EhlersRoofingFilterA(
    df: pd.DataFrame,
    hp_length: int = 80,
    lp_length: int = 40,
    arrow_distance: float = 100.0,
    point: float = 1.0
) -> pd.DataFrame:
    """Return all lines of the Ehlers Roofing Filter A as columns aligned to df.index.
    Columns: ['rfilt','trigger','hp','up','down'].
    Vectorized where possible; recursion computed with a tight NumPy loop. NaNs at warmup.
    """
    if hp_length <= 0 or lp_length <= 0:
        raise ValueError("hp_length and lp_length must be positive integers.")

    close = df['Close'].astype(float).to_numpy()
    n = close.size

    # Constants per Ehlers formulation
    twoPiPrd = np.sqrt(0.5) * 2.0 * np.pi / float(hp_length)
    a1 = (np.cos(twoPiPrd) + np.sin(twoPiPrd) - 1.0) / np.cos(twoPiPrd)

    a2 = np.exp(-np.sqrt(2.0) * np.pi / float(lp_length))
    beta = 2.0 * a2 * np.cos(np.sqrt(2.0) * np.pi / float(lp_length))
    c2 = beta
    c3 = -a2 * a2
    c1 = 1.0 - c2 - c3

    t1 = (1.0 - (a1 / 2.0)) ** 2
    t2 = 2.0 * (1.0 - a1)
    t3 = (1.0 - a1) ** 2

    # Allocate output arrays
    hp_arr = np.full(n, np.nan, dtype=float)
    rb_arr = np.full(n, np.nan, dtype=float)  # roofing filter (rfilt)
    tb_arr = np.full(n, np.nan, dtype=float)  # trigger

    # Internal state (zero-initialized as in original)
    h1 = 0.0  # hp[i-1]
    h2 = 0.0  # hp[i-2]
    r1 = 0.0  # rb[i-1]
    r2 = 0.0  # rb[i-2]

    for i in range(n):
        if i < 2:
            # Not enough history to compute second-order recursions; keep NaNs
            continue

        # High-pass filter (second-order recursive)
        h = t1 * (close[i] - 2.0 * close[i - 1] + close[i - 2]) + t2 * h1 - t3 * h2
        hp_arr[i] = h

        # Roofing filter (low-pass on HP) and trigger
        r = c1 * ((h + h1) / 2.0) + c2 * r1 + c3 * r2
        rb_arr[i] = r
        tb_arr[i] = r2  # trigger is rb lagged by 2

        # Update states
        h2, h1 = h1, h
        r2, r1 = r1, r

    # Build Series
    idx = df.index
    s_hp = pd.Series(hp_arr, index=idx, name='hp')
    s_rb = pd.Series(rb_arr, index=idx, name='rfilt')
    s_tb = pd.Series(tb_arr, index=idx, name='trigger')

    # Arrows (signals placed one bar after the crossover, per original)
    dst = float(arrow_distance) * float(point)
    cross_up = (s_rb.shift(1) > s_tb.shift(1)) & (s_rb.shift(2) < s_tb.shift(2))
    cross_dn = (s_rb.shift(1) < s_tb.shift(1)) & (s_rb.shift(2) > s_tb.shift(2))

    up_vals = np.where(cross_up.to_numpy(), s_rb.to_numpy() - dst, np.nan)
    dn_vals = np.where(cross_dn.to_numpy(), s_rb.to_numpy() + dst, np.nan)

    s_up = pd.Series(up_vals, index=idx, name='up')
    s_dn = pd.Series(dn_vals, index=idx, name='down')

    return pd.DataFrame(
        {
            'rfilt': s_rb,
            'trigger': s_tb,
            'hp': s_hp,
            'up': s_up,
            'down': s_dn,
        },
        index=idx,
    )