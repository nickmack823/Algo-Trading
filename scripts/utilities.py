import numpy as np
import pandas as pd


def convert_np_to_pd(
    indicator_output: pd.DataFrame | pd.Series | tuple | np.ndarray,
    index: pd.Index | None = None,
) -> pd.DataFrame | pd.Series:
    """
    Convert common indicator outputs (Series/DataFrame/ndarray/tuple of arrays) into
    a pandas object. If `index` is provided, align the result to that index and
    validate lengths. Does not apply any indicator-specific naming.
    """
    # Pass-through for pandas objects (with optional index alignment)
    if isinstance(indicator_output, pd.DataFrame):
        df = indicator_output.copy()
        if index is not None:
            if df.shape[0] == len(index):
                df.index = index
            elif df.shape[1] == len(index):
                df = df.T
                df.index = index
            else:
                raise ValueError(
                    f"DataFrame shape {df.shape} not aligned to index of length {len(index)}"
                )
        return df

    if isinstance(indicator_output, pd.Series):
        s = indicator_output.copy()
        if index is not None:
            if len(s) != len(index):
                raise ValueError(
                    f"Series length {len(s)} does not match index length {len(index)}"
                )
            s = pd.Series(s.to_numpy(), index=index, name=s.name)
        return s

    # Numpy array → Series/DataFrame, with optional index alignment
    if isinstance(indicator_output, np.ndarray):
        arr = np.asarray(indicator_output)
        if arr.ndim == 1:
            if index is not None and len(arr) != len(index):
                raise ValueError(
                    f"Array length {len(arr)} does not match index length {len(index)}"
                )
            return pd.Series(arr, index=index)
        if arr.ndim == 2:
            if index is None:
                return pd.DataFrame(arr)
            if arr.shape[0] == len(index):
                return pd.DataFrame(arr, index=index)
            if arr.shape[1] == len(index):
                return pd.DataFrame(arr.T, index=index)
            raise ValueError(
                f"Array shape {arr.shape} not aligned to index length {len(index)}"
            )
        raise TypeError(
            f"Unsupported ndarray with ndim={arr.ndim}; expected 1D or 2D array"
        )

    # Tuple/list of components → concatenate as columns (multi-output indicators)
    if isinstance(indicator_output, tuple):
        converted = []
        for i, item in enumerate(indicator_output):
            if isinstance(item, pd.Series):
                s = item
                if index is not None:
                    if len(s) != len(index):
                        raise ValueError(
                            f"Tuple component {i} length {len(s)} != index length {len(index)}"
                        )
                    s = pd.Series(s.to_numpy(), index=index, name=s.name)
                else:
                    s = s.reset_index(drop=True)
                converted.append(s)
            elif isinstance(item, np.ndarray):
                arr = np.asarray(item)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                if index is not None and len(arr) != len(index):
                    raise ValueError(
                        f"Tuple component {i} length {len(arr)} != index length {len(index)}"
                    )
                converted.append(pd.Series(arr, index=index))
            else:
                raise TypeError(
                    f"Unsupported type in tuple at index {i}: {type(item)}"
                )

        return pd.concat(converted, axis=1)

    raise TypeError(f"Unsupported type for conversion: {type(indicator_output)}")


def seconds_to_dhms_str(seconds):
    """Convert seconds to days, hours, minutes, seconds string"""
    days = int(seconds // (3600 * 24))
    hours = int((seconds // 3600) % 24)
    minutes = int((seconds // 60) % 60)
    seconds = int(seconds % 60)

    if days < 1:
        if hours < 1:
            if minutes < 1:
                return f"{seconds}s"
            return f"{minutes}m {seconds}s"
        return f"{hours}h {minutes}m {seconds}s"

    return f"""{days}d {hours}h {minutes}m {seconds}s"""
