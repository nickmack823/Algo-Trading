import numpy as np
import pandas as pd

def SecondOrderGaussianHighPassFilterMtfZones(
    df: pd.DataFrame,
    alpha: float = 0.14,
    timeframe: str | None = None,
    interpolate: bool = True,
    maxbars: int = 2000,
) -> pd.Series:
    """Return the primary indicator line aligned to df.index; vectorized where possible, handles NaNs, stable defaults.
    
    Parameters
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume'].
    - alpha: smoothing coefficient (MQL default 0.14).
    - timeframe: pandas offset alias (e.g., '15T') for multi-timeframe; None to use native timeframe.
    - interpolate: when timeframe is not None, linearly interpolate HTF values to LTF bars if True; else step (ffill).
    - maxbars: compute last N bars (like MQL maxbars); earlier values are NaN.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype=float, name="HPF")
    if "Timestamp" not in df.columns or "Close" not in df.columns:
        raise ValueError("df must contain 'Timestamp' and 'Close' columns")

    # Work on a time-ascending copy for consistent resampling/processing
    ts = pd.to_datetime(df["Timestamp"])
    asc_idx = np.argsort(ts.values.astype("datetime64[ns]"))
    df_asc = df.iloc[asc_idx].copy()
    df_asc["Timestamp"] = pd.to_datetime(df_asc["Timestamp"])
    dt_index = pd.DatetimeIndex(df_asc["Timestamp"])

    def _hpf_from_close(close_arr: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
        n = close_arr.shape[0]
        # Reverse so index 0 is the most recent (MQL-style), then compute backward recursion
        rev = close_arr[::-1].astype(float)
        # Fill NaNs to avoid propagating through recursion; then restore NaNs after computation
        rev_series = pd.Series(rev)
        rev_filled = rev_series.ffill().bfill().to_numpy()
        s = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            c0 = rev_filled[i]
            c1 = rev_filled[i + 1] if (i + 1) < n else 0.0
            c2 = rev_filled[i + 2] if (i + 2) < n else 0.0
            s1 = s[i + 1] if (i + 1) < n else 0.0
            s2 = s[i + 2] if (i + 2) < n else 0.0
            s[i] = a * (c0 - 2.0 * c1 + c2) + b * s1 - d * s2
        out = s[::-1]
        # Restore NaNs where original close was NaN
        nan_mask = ~np.isfinite(close_arr)
        if nan_mask.any():
            out[nan_mask] = np.nan
        return out

    a = (1.0 - alpha / 2.0) ** 2
    b = 2.0 * (1.0 - alpha)
    d = (1.0 - alpha) ** 2

    if timeframe is None:
        # Compute on native timeframe
        close_asc = df_asc["Close"].to_numpy(dtype=float)
        n_total = close_asc.shape[0]
        n_use = min(maxbars if maxbars is not None else n_total, n_total)
        hpf_full = np.full(n_total, np.nan, dtype=float)
        if n_use > 0:
            hpf_tail = _hpf_from_close(close_asc[-n_use:], a, b, d)
            hpf_full[-n_use:] = hpf_tail
        hpf_asc = pd.Series(hpf_full, index=df_asc.index, name="HPF")
    else:
        # Multi-timeframe: resample Close to HTF, compute HPF there, then align to LTF
        close_htf = pd.Series(df_asc["Close"].to_numpy(dtype=float), index=dt_index)
        close_htf = close_htf.resample(timeframe).last().dropna()
        if close_htf.empty:
            hpf_asc = pd.Series(np.nan, index=df_asc.index, name="HPF")
        else:
            hpf_vals_htf = _hpf_from_close(close_htf.to_numpy(), a, b, d)
            hpf_htf = pd.Series(hpf_vals_htf, index=close_htf.index, name="HPF")
            if interpolate:
                union_idx = hpf_htf.index.union(dt_index)
                aligned = (
                    hpf_htf.reindex(union_idx)
                    .sort_index()
                    .interpolate(method="time")
                    .reindex(dt_index)
                )
            else:
                aligned = hpf_htf.reindex(dt_index, method="ffill")
            # Apply maxbars on LTF alignment
            n_total = aligned.shape[0]
            n_use = min(maxbars if maxbars is not None else n_total, n_total)
            hpf_arr = aligned.to_numpy()
            if n_use < n_total:
                hpf_arr[: n_total - n_use] = np.nan
            hpf_asc = pd.Series(hpf_arr, index=df_asc.index, name="HPF")

    # Map back to original df.index order
    inv_order = np.empty_like(asc_idx)
    inv_order[asc_idx] = np.arange(len(asc_idx))
    hpf_out = hpf_asc.reindex(df_asc.index).to_numpy()
    hpf_out = hpf_out[inv_order]

    return pd.Series(hpf_out, index=df.index, name="HPF")