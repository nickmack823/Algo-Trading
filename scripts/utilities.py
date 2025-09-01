import numpy as np
import pandas as pd


def convert_np_to_pd(
    indicator_output: pd.DataFrame | pd.Series | tuple | np.ndarray,
) -> pd.DataFrame | pd.Series:
    if isinstance(indicator_output, pd.DataFrame) or isinstance(
        indicator_output, pd.Series
    ):
        return indicator_output
    elif isinstance(indicator_output, np.ndarray):
        return pd.Series(indicator_output)
    elif isinstance(indicator_output, tuple):
        # If it's a tuple of Series or arrays, build a DataFrame with auto-named columns
        converted = []

        for i, item in enumerate(indicator_output):
            if isinstance(item, pd.Series):
                converted.append(item.reset_index(drop=True))
            elif isinstance(item, np.ndarray):
                converted.append(pd.Series(item))
            else:
                raise TypeError(f"Unsupported type in tuple at index {i}: {type(item)}")

        return pd.concat(converted, axis=1)
    else:
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
