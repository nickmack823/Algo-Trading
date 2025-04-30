import numpy as np
import pandas as pd

# Helper Functions

# def STDEV(price_data: pd.DataFrame, baseline: str, window: int, threshold: float, bound: str) -> pd.Series:
#     """ Calculate the Standard Deviation of the given indicator.

#     Args:
#         price_data (pd.DataFrame): A Pandas DataFrame of float values representing price data with
#             columns 'Open', 'High', 'Low', 'Close', 'Volume'.
#         baseline (str): The name of the indicator to calculate the standard deviation of.
#         window (int): The number of periods to use for the rolling window.
#         threshold (float): The number of standard deviations to use as the threshold.
#         bound (str): The bound to use for the standard deviation calculation. Can be 'upper' or 'lower'.
#     Returns:
#         pd.Series: A Pandas Series representing the standard deviation values.
#     """
#     indicator_function = baselines[baseline]['function']
#     indicator_values = indicator_function(price_data, **baselines[baseline]['default_params'])
#     stdev = indicator_values.rolling(window).std() * threshold

#     if bound == 'upper':
#         price_thresholds = indicator_values.add(stdev, axis=0)
#     else:
#         price_thresholds = indicator_values.sub(stdev, axis=0)

#     return price_thresholds


def series_wma(series, period):
    weights = np.arange(1, period + 1)
    wma = series.rolling(window=period).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
    )
    return wma


def series_ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def zero_lag_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Zero-Lag Exponential Moving Average (ZLEMA) for the given series and period."""
    ema = series.ewm(span=period).mean()
    zlema = (2 * ema) - ema.ewm(span=period).mean()
    return zlema


def j_tpo_value(input_prices, period, shift):
    value = 0
    arr1 = [0] + list(input_prices[shift : shift + period][::-1]) + [0]
    arr2 = [0] + list(range(1, period + 1)) + [0]
    arr3 = arr2.copy()

    for m in range(1, period):
        maxval = arr1[m]
        maxloc = m

        for j in range(m + 1, period + 1):
            if arr1[j] < maxval:
                maxval = arr1[j]
                maxloc = j

        arr1[m], arr1[maxloc] = arr1[maxloc], arr1[m]
        arr2[m], arr2[maxloc] = arr2[maxloc], arr2[m]

    m = 1
    while m < period:
        j = m + 1
        flag = True
        accum = arr3[m]

        while flag:
            if arr1[m] != arr1[j]:
                if (j - m) > 1:
                    accum = accum / (j - m)
                    for n in range(m, j):
                        arr3[n] = accum
                flag = False
            else:
                accum += arr3[j]
                j += 1

        m = j

    normalization = 12.0 / (period * (period - 1) * (period + 1))
    lenp1half = (period + 1) * 0.5

    for m in range(1, period + 1):
        value += (arr3[m] - lenp1half) * (arr2[m] - lenp1half)

    value = normalization * value

    return value


# ATR Functions


def ATR(price_data: pd.DataFrame, period: int) -> pd.Series:
    high_low = price_data["High"] - price_data["Low"]
    high_close = np.abs(price_data["High"] - price_data["Close"].shift())
    low_close = np.abs(price_data["Low"] - price_data["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)

    true_range = np.max(ranges, axis=1)

    atr = true_range.rolling(period).sum() / period
    return atr


def FilteredATR(
    price_data: pd.DataFrame, period: int = 34, ma_period: int = 34, ma_shift: int = 0
) -> pd.Series:
    """
    Calculate the Filtered ATR indicator for the given price data.

    Parameters:
    price_data (pd.DataFrame): OHLCV price data
    period (int, optional): ATR period, default is 34
    ma_period (int, optional): MA period, default is 34
    ma_shift (int, optional): Moving average shift, default is 0

    Returns:
    pd.Series: Filtered ATR values
    """
    # Calculate the True Range
    tr = pd.Series(
        np.maximum(
            price_data["High"] - price_data["Low"],
            np.maximum(
                np.abs(price_data["High"] - price_data["Close"].shift(1)),
                np.abs(price_data["Low"] - price_data["Close"].shift(1)),
            ),
            np.abs(price_data["High"] - price_data["Close"].shift(1)),
        ),
        name="TrueRange",
    )

    # Calculate the ATR
    atr = tr.rolling(window=period).mean()

    # Calculate the moving average on ATR
    atr_ma = atr.ewm(span=ma_period, adjust=False).mean()

    # Shift the moving average (if required)
    if ma_shift != 0:
        atr_ma = atr_ma.shift(ma_shift)

    # Return the calculated Filtered ATR values
    return atr_ma


# Baseline Functions


def SMA(price_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    return pd.Series(price_data.Close).rolling(period).mean()


def EMA(price_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates Exponential Moving Average (EMA) for a given period and column of a pandas DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing the data to compute EMA for.
    period : int
        The period to use for computing EMA.

    Returns:
    --------
    ema : pandas Series
        A pandas Series containing the computed EMA values.
    """
    ema = price_data["Close"].ewm(span=period, adjust=False).mean()

    return ema


def WMA(price_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Weighted Moving Average (WMA) using a Pandas DataFrame of OHLCV data.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame containing OHLCV data.
        period (int): The period for the WMA calculation.

    Returns:
        _type_: A Pandas Series containing the WMA values.
    """
    price_series = price_data["Close"]
    weights = np.arange(1, period + 1)

    return price_series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def HMA(price_data: pd.DataFrame, period: int = 13) -> pd.Series:
    """Calculate the Hull Moving Average (HMA) using a Pandas DataFrame of OHLCV data.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame containing OHLCV data.
        period (int): The period for the HMA calculation.

    Returns:
        pd.Series: A Pandas Series containing the HMA values.
    """
    close_series = price_data["Close"]
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))

    wma_half = series_wma(close_series, half_length)
    wma_full = series_wma(close_series, period)

    hma_series = 2 * wma_half - wma_full

    return series_wma(hma_series, sqrt_length)


def VIDYA(price_data: pd.DataFrame, period: int = 9, histper: int = 30) -> pd.Series:
    vidya = pd.Series(index=price_data.index, dtype="float64")

    def iStdDev(series, length):
        return series.rolling(window=length).std()

    # Calculate standard deviation values only once
    std_dev_period = iStdDev(price_data["Close"], period)
    std_dev_histper = iStdDev(price_data["Close"], histper)

    i = len(price_data) - 1
    while i >= 0:
        if i < len(price_data) - histper:
            k = std_dev_period.iloc[i] / std_dev_histper.iloc[i]
            sc = 2.0 / (period + 1)
            vidya.iloc[i] = (
                k * sc * price_data["Close"].iloc[i] + (1 - k * sc) * vidya.iloc[i + 1]
            )
        else:
            vidya.iloc[i] = price_data["Close"].iloc[i]

        i -= 1

    return vidya


def KAMA(
    price_data: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30
) -> pd.Series:
    """Calculate the Kaufman's Adaptive Moving Average (KAMA) using a Pandas DataFrame of OHLCV data.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame containing OHLCV data.
        period (int, optional): The period to calculate for. Defaults to 10.
        fast (int, optional): The period of the fast EMA. Defaults to 2.
        slow (int, optional): The period of the slow EMA. Defaults to 30.

    Returns:
        pd.Series: A Pandas Series containing the KAMA values.
    """
    close = price_data["Close"]

    # Calculate the absolute price change
    price_change = np.abs(close - close.shift())

    # Calculate the volatility
    volatility = price_change.rolling(window=period).sum()

    # Calculate the Efficiency Ratio (ER)
    er = price_change / volatility

    # Calculate the Smoothing Constant (SC)
    fast_weight = 2 / (fast + 1)
    slow_weight = 2 / (slow + 1)
    sc = np.square(er * (fast_weight - slow_weight) + slow_weight)

    # Calculate KAMA
    kama = pd.Series(index=close.index, dtype="float64")
    kama.iloc[period - 1] = close.iloc[period - 1]

    for i in range(period, len(close)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            close.iloc[i] - kama.iloc[i - 1]
        )

    return kama


def ALMA(
    price_data: pd.DataFrame, period: int = 9, sigma: int = 6, offset: float = 0.85
) -> pd.Series:
    """
    Calculate the Arnaud Legoux Moving Average (ALMA) for the given price data.

    Parameters:
    price_data (DataFrame): A DataFrame containing OHLCV price data with columns 'Open', 'High', 'Low', 'Close', and 'Volume'.
    window (int, optional): The window size to use for the moving average calculation. Defaults to 9.
    sigma (int, optional): The sigma value for Gaussian distribution. Defaults to 6.
    offset (float, optional): The offset value for the Gaussian distribution center. Defaults to 0.85.

    Returns:
    Series: A pandas Series containing the ALMA values.
    """

    # Get the Close prices from the price data
    close = price_data["Close"]

    # Calculate the center of the Gaussian distribution
    m = int(offset * (period - 1))

    # Calculate the standard deviation for the Gaussian distribution
    s = period / sigma

    # Initialize an array of zeros with the size of the window for the ALMA weights
    alma_weights = np.zeros(period)

    # Calculate the weights for the Gaussian distribution function
    for i in range(period):
        xi = i - m
        alma_weights[i] = np.exp(-(xi**2) / (2 * s**2))

    # Normalize the ALMA weights
    alma_weights /= np.sum(alma_weights)

    # Calculate the ALMA values using a centered rolling window and apply the weights
    return close.rolling(window=period, center=False).apply(
        lambda x: np.sum(alma_weights * x), raw=True
    )


def T3(price_data: pd.DataFrame, period: int = 5, vfactor: float = 0.7) -> pd.Series:
    """
    Calculate the T3 moving average indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    period : int, optional
        The number of periods used for the calculation, by default 5.
    vfactor : float, optional
        The volume factor used for smoothing, by default 0.7.

    Returns
    -------
    pd.Series
        A Series containing the T3 values.
    """
    # Calculate the T3 components
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]
    typical_price = (high + low + close) / 3
    e1 = series_ema(typical_price, period)
    e2 = series_ema(e1, period)
    e3 = series_ema(e2, period)
    e4 = series_ema(e3, period)
    e5 = series_ema(e4, period)
    e6 = series_ema(e5, period)

    # Calculate the T3
    c1 = -vfactor * vfactor * vfactor
    c2 = 3 * vfactor * vfactor + 3 * vfactor * vfactor * vfactor
    c3 = -6 * vfactor * vfactor - 3 * vfactor - 3 * vfactor * vfactor * vfactor
    c4 = 1 + 3 * vfactor + vfactor * vfactor * vfactor + 3 * vfactor * vfactor
    T3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    return T3


def FantailVMA(
    price_data: pd.DataFrame,
    adx_length: int = 2,
    weighting: float = 2.0,
    ma_length: int = 1,
) -> pd.DataFrame:
    """
    Calculate the Fantail Volume-Weighted Moving Average (VMA) indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    adx_length : int, optional
        The number of periods used for the ADX calculation, by default 2.
    weighting : float, optional
        The weighting factor, by default 2.0.
    ma_length : int, optional
        The number of periods used for the moving average calculation, by default 1.

    Returns
    -------
    DataFrame
        A DataFrame containing the Fantail VMA values.
    """
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]

    n = len(close)

    # Initialize arrays
    spdi = np.zeros(n)
    smdi = np.zeros(n)
    str_ = np.zeros(n)
    adx = np.zeros(n)
    varma = np.zeros(n)
    ma = np.zeros(n)

    # Calculate the Fantail VMA
    for i in range(n - 2, -1, -1):
        hi = high[i]
        hi1 = high[i + 1]
        lo = low[i]
        lo1 = low[i + 1]
        close1 = close[i + 1]

        bulls = 0.5 * (abs(hi - hi1) + (hi - hi1))
        bears = 0.5 * (abs(lo1 - lo) + (lo1 - lo))

        if bulls > bears:
            bears = 0
        elif bulls < bears:
            bulls = 0
        else:
            bulls = 0
            bears = 0

        spdi[i] = (weighting * spdi[i + 1] + bulls) / (weighting + 1)
        smdi[i] = (weighting * smdi[i + 1] + bears) / (weighting + 1)

        tr = max(hi - lo, hi - close1)
        str_[i] = (weighting * str_[i + 1] + tr) / (weighting + 1)

        if str_[i] > 0:
            pdi = spdi[i] / str_[i]
            mdi = smdi[i] / str_[i]
        else:
            pdi = mdi = 0

        if (pdi + mdi) > 0:
            dx = abs(pdi - mdi) / (pdi + mdi)
        else:
            dx = 0

        adx[i] = (weighting * adx[i + 1] + dx) / (weighting + 1)
        vadx = adx[i]

        adxmin = min(adx[i : i + adx_length])
        adxmax = max(adx[i : i + adx_length])

        diff = adxmax - adxmin
        const = (vadx - adxmin) / diff if diff > 0 else 0

        varma[i] = ((2 - const) * varma[i + 1] + const * close[i]) / 2

    # Calculate the MA
    ma = pd.Series(varma).rolling(window=ma_length).mean()

    # Convert arrays to pandas Series
    varma_series = pd.Series(varma)
    ma_series = pd.Series(ma)

    return pd.DataFrame({"Fantail_VMA": varma_series, "Fantail_MA": ma_series})


def EHLERS(price_data: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Calculate the Ehler's 2 Pole Super Smoother Filter.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    period : int, optional
        The number of periods used for the filter calculation, by default 10.

    Returns
    -------
    pd.Series
        A Series containing the Ehler's 2 Pole Super Smoother Filter values.
    """
    series = price_data["Close"]

    n = len(series)

    # Calculate the Ehler's 2 Pole Super Smoother Filter
    a = np.exp(-1.414 * np.pi / period)
    b = 2 * a * np.cos(1.414 * np.pi / period)
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3

    output = np.zeros(n)
    for i in range(2, n):
        output[i] = (
            c1 * (series[i] + series[i - 1]) / 2
            + c2 * output[i - 1]
            + c3 * output[i - 2]
        )

    return pd.Series(output)


def McGinleyDI(
    price_data: pd.DataFrame, period: int = 12, mcg_constant: float = 5
) -> pd.Series:
    """
    Calculate the McGinley Dynamic Indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    period : int, optional
        The number of periods used for the indicator calculation, by default 12.
    mcg_constant : float, optional
        The McGinley constant, by default 5.

    Returns
    -------
    pd.Series
        A Series containing the McGinley Dynamic Indicator values.
    """
    series = price_data["Close"]

    n = len(series)
    mcg = np.zeros(n)

    # Calculate the Simple Moving Average for the given period
    ma = series.rolling(window=period).mean()

    for i in range(period, n):
        price = series[i]
        mcg[i] = ma[i - 1] + (price - ma[i - 1]) / (
            mcg_constant * period * np.power(price / ma[i - 1], 4)
        )

    return mcg


def DEMA(price_data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Double Exponential Moving Average (DEMA) of the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    period : int, optional, default: 14
        The number of periods to use in the calculation of the DEMA.

    Returns
    -------
    pd.Series
        A Series containing the DEMA values for the specified price data and period.
    """
    # Calculate the Exponential Moving Average (EMA) for the given period
    ema1 = price_data["Close"].ewm(span=period).mean()

    # Calculate the EMA of the first EMA
    ema2 = ema1.ewm(span=period).mean()

    # Calculate the DEMA using the three EMAs
    dema = 2 * ema1 - ema2

    # Return the DEMA
    return dema


def TEMA(price_data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Triple Exponential Moving Average (TEMA) of the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    period : int, optional, default: 14
        The number of periods to use in the calculation of the TEMA.

    Returns
    -------
    pd.Series
        A Series containing the TEMA values for the specified price data and period.
    """
    # Calculate the Exponential Moving Average (EMA) for the given period
    ema1 = price_data["Close"].ewm(span=period).mean()

    # Calculate the EMA of the first EMA
    ema2 = ema1.ewm(span=period).mean()

    # Calculate the EMA of the second EMA
    ema3 = ema2.ewm(span=period).mean()

    # Calculate the TEMA using the three EMAs
    tema = 3 * ema1 - 3 * ema2 + ema3

    # Return the TEMA
    return tema


def KijunSen(df: pd.DataFrame, period: int = 26, shift: int = 9) -> pd.Series:
    """
    Calculate Ichimoku Kinko Hyo Kijun-Sen line

    Parameters:
    df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns
    kijun (int): Kijun period
    kijun_shift (int): Kijun shift

    Returns:
    pd.Series: Series with Kijun-Sen line values
    """
    high = df["High"]
    low = df["Low"]
    kijun_buffer = [np.nan] * len(df)

    # Calculate Kijun-Sen for main part
    for i in range(period, len(df)):
        kijun_buffer[i - shift] = (
            high.iloc[i - period : i].max() + low.iloc[i - period : i].min()
        ) / 2

    # Calculate Kijun-Sen for initial part
    for i in range(shift - 1, -1, -1):
        kijun_buffer[i] = (
            high.iloc[: period - shift + i].max() + low.iloc[: period - shift + i].min()
        ) / 2

    return pd.Series(kijun_buffer, index=df.index)


# Confirmation Functions


def KASE(
    price_data: pd.DataFrame,
    pstLength: int = 9,
    pstX: int = 5,
    pstSmooth: int = 3,
    smoothPeriod: int = 10,
) -> pd.DataFrame:
    """Calculate the Kase Permission Stochastic Smoothed (KPSS) using a Pandas DataFrame.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame containing OHLCV data.
        pstLength (int, optional): The period for the Permission Stochastic calculation. Defaults to 9.
        pstX (int, optional): . Defaults to 5.
        pstSmooth (int, optional): . Defaults to 3.
        smoothPeriod (int, optional): . Defaults to 10.

    Returns:
        _type_: _description_
    """
    lookBackPeriod = pstLength * pstX
    alpha = 2.0 / (1.0 + pstSmooth)

    TripleK, TripleDF, TripleDS, TripleDSs, TripleDFs = 0.0, 0.0, 0.0, 0.0, 0.0

    fmin = price_data["Low"].rolling(window=lookBackPeriod).min()
    fmax = price_data["High"].rolling(window=lookBackPeriod).max() - fmin

    TripleK = 100.0 * (price_data["Close"] - fmin) / fmax
    TripleK = TripleK.replace([np.inf, -np.inf], 0).fillna(0)

    TripleDF = TripleK.ewm(alpha=alpha, adjust=False).mean().shift(pstX)
    TripleDS = (TripleDF.shift(pstX) * 2.0 + TripleDF) / 3.0

    def hma(series, period):
        half_length = period // 2
        sqrt_length = int(np.sqrt(period))

        wma_half = series_wma(series, half_length)
        wma_full = series_wma(series, period)

        hma_series = 2 * wma_half - wma_full

        return series_wma(hma_series, sqrt_length)

    TripleDSs = TripleDS.rolling(window=3).mean()
    pssBuffer = hma(TripleDSs, smoothPeriod)

    TripleDFs = TripleDF.rolling(window=3).mean()
    pstBuffer = hma(TripleDFs, smoothPeriod)

    # pst > pss == buy
    # pst < pss == sell

    # Create a Pandas DataFrame to store the KPSS lines
    result = pd.DataFrame(
        {
            "KPSS_BUY": pstBuffer,
            "KPSS_SELL": pssBuffer,
        }
    )

    return result


def MACDZeroLag(
    price_data: pd.DataFrame,
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate the Zero-Lag MACD and signal line.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    short_period : int, optional
        The number of periods for the short-term EMA, by default 12.
    long_period : int, optional
        The number of periods for the long-term EMA, by default 26.
    signal_period : int, optional
        The number of periods for the signal line EMA, by default 9.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the MACD and signal line values.
    """
    close = price_data["Close"]

    short_zlema = zero_lag_ema(close, short_period)
    long_zlema = zero_lag_ema(close, long_period)

    macd_line = short_zlema - long_zlema
    signal_line = zero_lag_ema(macd_line, signal_period)

    result = pd.DataFrame(
        {
            "MACD": macd_line,
            "SIGNAL": signal_line,
        }
    )

    return result


def KalmanFilter(
    price_data: pd.DataFrame, k: float = 1, sharpness: float = 1
) -> pd.DataFrame:
    """
    Calculate the Kalman Filter forex indicator based on the provided price data using the 'Close' prices.

    Parameters:
    price_data (pd.DataFrame): A Pandas DataFrame of OHLCV price data.
    k (float): K parameter for the Kalman Filter, default is 1.
    sharpness (float): Sharpness parameter for the Kalman Filter, default is 1.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the calculated Kalman Filter values.
    """

    ext_map_buffer_up = np.empty(len(price_data))
    ext_map_buffer_up[:] = np.nan
    ext_map_buffer_down = np.empty(len(price_data))
    ext_map_buffer_down[:] = np.nan

    velocity = 0
    distance = 0
    error = 0
    value = price_data["Close"].iloc[1]

    for i in range(len(price_data) - 1, -1, -1):
        price = price_data["Close"].iloc[i]
        distance = price - value
        error = value + distance * np.sqrt(sharpness * k / 100)
        velocity = velocity + distance * k / 100
        value = error + velocity

        if velocity > 0:
            ext_map_buffer_up[i] = value
            ext_map_buffer_down[i] = np.nan

            if i < len(price_data) - 1 and np.isnan(ext_map_buffer_up[i + 1]):
                ext_map_buffer_up[i + 1] = ext_map_buffer_down[i + 1]
        else:
            ext_map_buffer_up[i] = np.nan
            ext_map_buffer_down[i] = value

            if i < len(price_data) - 1 and np.isnan(ext_map_buffer_down[i + 1]):
                ext_map_buffer_down[i + 1] = ext_map_buffer_up[i + 1]

    result = pd.DataFrame({"Up": ext_map_buffer_up, "Down": ext_map_buffer_down})
    return result


def Fisher(
    price_data: pd.DataFrame,
    range_periods: int = 10,
    price_smoothing: float = 0.3,
    index_smoothing: float = 0.3,
) -> pd.DataFrame:
    """
    Calculate the Fisher Indicator for the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    range_periods : int, optional, default: 10
        The number of periods to use in the calculation of the Fisher Indicator.
    price_smoothing : float, optional, default: 0.3
        The price smoothing factor.
    index_smoothing : float, optional, default: 0.3
        The index smoothing factor.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Fisher Indicator values for the specified price data and parameters.
    """
    # Calculate mid-price
    mid_price = (price_data["High"] + price_data["Low"]) / 2

    # Calculate the highest high and lowest low for the given range_periods
    highest_high = price_data["High"].rolling(window=range_periods).max()
    lowest_low = price_data["Low"].rolling(window=range_periods).min()

    # Calculate the greatest range and avoid division by zero
    greatest_range = (highest_high - lowest_low).replace(0, 0.1 * 10**-5)

    # Calculate the price location in the current range
    price_location = 2 * ((mid_price - lowest_low) / greatest_range) - 1

    # Apply price smoothing
    smoothed_location = price_location.ewm(alpha=(1 - price_smoothing)).mean()

    # Limit smoothed_location between -0.99 and 0.99 to avoid infinite values in the logarithm
    smoothed_location = smoothed_location.clip(lower=-0.99, upper=0.99)

    # Calculate the Fisher Index
    fisher_index = np.log((1 + smoothed_location) / (1 - smoothed_location))

    # Apply index smoothing
    smoothed_fisher = fisher_index.ewm(alpha=(1 - index_smoothing)).mean()

    # Separate uptrend and downtrend values
    uptrend = smoothed_fisher.where(smoothed_fisher > 0, 0)
    downtrend = smoothed_fisher.where(smoothed_fisher <= 0, 0)

    # Return the Fisher Indicator values as a DataFrame
    return pd.DataFrame({"Fisher": smoothed_fisher})


def BullsBearsImpulse(price_data: pd.DataFrame, ma_period: int = 13) -> pd.DataFrame:
    """
    Calculate the Bears Bulls Impulse indicator for the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    ma_period : int, optional, default: 13
        The number of periods to use in the calculation of the moving average.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Bears Bulls Impulse indicator values for the specified price data and parameters.
    """
    ma = price_data["Close"].rolling(window=ma_period).mean()

    bulls = price_data["High"] - ma
    bears = price_data["Low"] - ma

    avg = bears + bulls

    buffer1 = (avg >= 0).replace({True: 1.0, False: -1.0})
    buffer2 = -buffer1

    return pd.DataFrame({"Bulls": buffer1, "Bears": buffer2})


def Gen3MA(
    price_data: pd.DataFrame, period: int = 220, sampling_period: int = 50
) -> pd.DataFrame:
    """
    Calculate the 3rd Generation Moving Average indicator for the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    ma_period : int, optional, default: 220
        The number of periods to use in the calculation of the 3GMA.
    ma_sampling_period : int, optional, default: 50
        The number of periods to use in the calculation of the crossing MA.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the 3rd Generation Moving Average indicator values for the specified price data and parameters.
    """
    ma1 = price_data["Close"].rolling(window=period).mean()
    ma2 = ma1.rolling(window=sampling_period).mean()

    alpha = sampling_period / period

    # 3rd Generation Moving Average line
    ma3g = (1 + alpha) * ma1 - alpha * ma2

    return pd.DataFrame({"MA3G": ma3g, "SignalMA": ma2})


def Aroon(price_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates the Aroon indicator.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    period (int): The number of periods to use in the calculation. Default is 14.

    Returns:
    pd.DataFrame: A Pandas DataFrame with two columns: 'Aroon Up' and 'Aroon Down'.
    """
    aroon_up = [np.nan] * period
    aroon_down = [np.nan] * period

    for i in range(period, len(price_data)):
        high_period = price_data["High"].iloc[i - period : i].tolist()
        low_period = price_data["Low"].iloc[i - period : i].tolist()
        n_high = high_period.index(max(high_period))
        n_low = low_period.index(min(low_period))

        aroon_up.append(100.0 * (period - n_high) / period)
        aroon_down.append(100.0 * (period - n_low) / period)

    return pd.DataFrame({"Aroon Up": aroon_up, "Aroon Down": aroon_down})


def Coral(price_data: pd.DataFrame, period: int = 34) -> pd.DataFrame:
    """
    Calculates the THV Coral indicator for the given price data and period.
    Returns a Pandas DataFrame containing the calculated indicator values.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    period (int): The period used in the indicator calculation. Default is 34.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the THV Coral indicator values.
    """
    gd_88 = 0.4
    g_ibuf_96 = np.empty(len(price_data))
    g_ibuf_100 = np.empty(len(price_data))
    g_ibuf_104 = np.empty(len(price_data))
    g_ibuf_108 = np.empty(len(price_data))
    gda_112 = np.empty(len(price_data))
    gda_116 = np.empty(len(price_data))
    gda_120 = np.empty(len(price_data))
    gda_124 = np.empty(len(price_data))
    gda_128 = np.empty(len(price_data))
    gda_132 = np.empty(len(price_data))
    gd_136 = -(gd_88**3)
    gd_144 = 3.0 * (gd_88**2 + gd_88**3)
    gd_152 = -3.0 * (2.0 * gd_88**2 + gd_88 + gd_88**3)
    gd_160 = 3.0 * gd_88 + 1.0 + gd_88**3 + 3.0 * gd_88**2
    gi_84 = period
    if gi_84 < 1:
        gi_84 = 1
    gi_84 = (gi_84 - 1) // 2 + 1
    gd_176 = 2 / (gi_84 + 1)
    gd_184 = 1 - gd_176

    for i in range(len(price_data)):
        if i == 0:
            gda_112[i] = price_data["Close"][i]
            gda_116[i] = gda_112[i]
            gda_120[i] = gda_116[i]
            gda_124[i] = gda_120[i]
            gda_128[i] = gda_124[i]
            gda_132[i] = gda_128[i]
        else:
            gda_112[i] = gd_176 * price_data["Close"][i] + gd_184 * gda_112[i - 1]
            gda_116[i] = gd_176 * gda_112[i] + gd_184 * gda_116[i - 1]
            gda_120[i] = gd_176 * gda_116[i] + gd_184 * gda_120[i - 1]
            gda_124[i] = gd_176 * gda_120[i] + gd_184 * gda_124[i - 1]
            gda_128[i] = gd_176 * gda_124[i] + gd_184 * gda_128[i - 1]
            gda_132[i] = gd_176 * gda_128[i] + gd_184 * gda_132[i - 1]

        g_ibuf_108[i] = (
            gd_136 * gda_132[i]
            + gd_144 * gda_128[i]
            + gd_152 * gda_124[i]
            + gd_160 * gda_120[i]
        )
        ld_0 = g_ibuf_108[i]
        if i == 0:
            ld_8 = g_ibuf_108[i + 1]
        else:
            ld_8 = g_ibuf_108[i - 1]
        g_ibuf_96[i] = ld_0
        g_ibuf_100[i] = ld_0
        g_ibuf_104[i] = ld_0
        if ld_8 > ld_0:
            g_ibuf_100[i] = np.nan
        elif ld_8 < ld_0:
            g_ibuf_104[i] = np.nan
        else:
            g_ibuf_96[i] = np.nan

    return pd.DataFrame({"THV Coral": g_ibuf_108, "Up": g_ibuf_100, "Down": g_ibuf_104})


def CenterOfGravity(price_data, period=10) -> pd.Series:
    """
    Calculates Ehler's Center of Gravity oscillator

    Args:
    price_data: Pandas DataFrame of OHLCV data
    period: int - period of the oscillator

    Returns:
    Pandas DataFrame of Center of Gravity oscillator values
    """

    def p(index: int) -> float:
        return (price_data["High"][index] + price_data["Low"][index]) / 2.0

    cg = []

    for s in range(len(price_data) - period):
        num = 0.0
        denom = 0.0
        for count in range(period):
            p_value = p(s + count)
            num += (1.0 + count) * p_value
            denom += p_value

        if denom != 0:
            cg_val = -num / denom + (period + 1.0) / 2.0
        else:
            cg_val = 0

        cg.append(cg_val)

    # Add NaN values to the end to match the length of the input price_data
    cg.extend([float("nan")] * period)

    return pd.Series(cg)


def GruchaIndex(
    price_data: pd.DataFrame, period: int = 10, ma_period: int = 10
) -> pd.DataFrame:
    """
    Calculates the Grucha Index and its moving average.

    Parameters:
    price_data (pd.DataFrame): DataFrame of OHLCV data.
    Okresy (int): Number of periods to calculate the Grucha Index.
    MA_Okresy (int): Number of periods to calculate the moving average of the Grucha Index.

    Returns:
    pd.DataFrame: DataFrame of the Grucha Index and its moving average.
    """
    ExtMapBuffer1 = [0.0] * len(price_data)
    tab = [0.0] * len(price_data)
    srednia = [0.0] * len(price_data)

    for i in range(len(price_data)):
        close = price_data["Close"][i]
        open = price_data["Open"][i]

        dResult = open - close
        tab[i] = dResult

        if i >= period - 1:
            gora = 0
            dol = 0
            for j in range(i, i - period, -1):
                if tab[j] < 0:
                    gora += tab[j]
                elif tab[j] >= 0:
                    dol -= tab[j]

            if dol <= 0:
                dol = dol * (-1)
            if gora <= 0:
                gora = gora * (-1)
            suma = dol + gora

            if suma == 0:
                wynik = 0
            elif suma > 0:
                wynik = (gora / suma) * 100
            else:
                wynik = 0

            ExtMapBuffer1[i] = wynik

    for i in range(len(price_data) - ma_period + 1):
        srednia[i + ma_period - 1] = sum(ExtMapBuffer1[i : i + ma_period]) / ma_period

    return pd.DataFrame({"Grucha Index": ExtMapBuffer1, "MA of Grucha Index": srednia})


def HalfTrend(price_data: pd.DataFrame, amplitude: int = 2) -> pd.DataFrame:
    def single_atr(index: int) -> float:
        return (price_data["High"][index] - price_data["Low"][index]) / 2

    nexttrend = False
    maxlowprice = 0
    minhighprice = float("inf")
    up = [0] * len(price_data)
    down = [0] * len(price_data)
    atrlo = [0] * len(price_data)
    atrhi = [0] * len(price_data)
    trend = [0] * len(price_data)

    for i in range(len(price_data) - 1, -1, -1):
        lowprice_i = price_data["Low"][i - amplitude : i].min()
        highprice_i = price_data["High"][i - amplitude : i].max()
        lowma = price_data["Low"][i - amplitude : i].mean()
        highma = price_data["High"][i - amplitude : i].mean()
        trend[i] = trend[i + 1] if i + 1 < len(price_data) else trend[i]
        atr_val = single_atr(i)

        if i + 1 < len(price_data):
            if nexttrend:
                maxlowprice = max(lowprice_i, maxlowprice)

                if (
                    highma < maxlowprice
                    and price_data["Close"][i] < price_data["Low"][i + 1]
                ):
                    trend[i] = 1.0
                    nexttrend = False
                    minhighprice = highprice_i
            else:
                minhighprice = min(highprice_i, minhighprice)

                if (
                    lowma > minhighprice
                    and price_data["Close"][i] > price_data["High"][i + 1]
                ):
                    trend[i] = 0.0
                    nexttrend = True
                    maxlowprice = lowprice_i

        if trend[i] == 0.0:
            if i + 1 < len(price_data) and trend[i + 1] != 0.0:
                up[i] = down[i + 1]
                up[i + 1] = up[i]
            else:
                up[i] = (
                    max(maxlowprice, up[i + 1]) if i + 1 < len(price_data) else up[i]
                )

            atrhi[i] = up[i] - atr_val
            atrlo[i] = up[i]
            down[i] = 0.0
        else:
            if i + 1 < len(price_data) and trend[i + 1] != 1.0:
                down[i] = up[i + 1]
                down[i + 1] = down[i]
            else:
                down[i] = (
                    min(minhighprice, down[i + 1])
                    if i + 1 < len(price_data)
                    else down[i]
                )

            atrhi[i] = down[i] + atr_val
            atrlo[i] = down[i]
            up[i] = 0.0

    return pd.DataFrame(
        {"Up": up, "Down": down}
    )  # , 'AtrLo': atrlo, 'AtrHi': atrhi, 'Trend': trend})


def J_TPO(price_data: pd.DataFrame, period: int = 14):
    """
    Calculates the J_TPO_Velocity indicator.

    J_TPO is an oscillator between -1 and +1, a nonparametric statistic quantifying how well the prices are ordered
    in consecutive ups (+1) or downs (-1) or intermediate cases in between. J_TPO_Velocity takes that value and
    multiplies it by the range, highest high to lowest low in the period (in pips), divided by the period length.
    Therefore, J_TPO_Velocity is a rough estimate of "velocity" as in "pips per bar". Positive of course means going
    up and negative means going down. J_TPO_Velocity thus crosses zero at exactly the same time as J_TPO, but the
    absolute magnitude is different.

    Args:
    price_data: Pandas DataFrame of OHLCV data.
    period: Length of the indicator. Default is 14.

    Returns:
    Pandas DataFrame of J_TPO_Velocity values.
    """
    close_prices = price_data["Close"]
    high_prices = price_data["High"]
    low_prices = price_data["Low"]
    # Default assumes 4-digit forex pairs; now, it's 5-digit for all non-JPY pairs
    # point = 0.0001 # Assuming a 4-digit forex pair
    # Count the number of digits after the decimal point
    point = 10 ** (
        -len(str(close_prices[0]).split(".")[1])
    )  # To normalize the indicator's value

    def j_tpo_range(high_prices, low_prices, period, shift):
        highest_high = high_prices[shift : shift + period].max()
        lowest_low = low_prices[shift : shift + period].min()

        return highest_high - lowest_low  # /point

    ext_map_buffer = [np.nan] * len(close_prices)

    if period < 3:
        # print("J_TPO_B: length must be at least 3")
        return ext_map_buffer

    for i in range(len(close_prices) - period):
        ext_map_buffer[i] = (
            j_tpo_value(close_prices, period, i)
            * j_tpo_range(high_prices, low_prices, period, i)
            / period
        )

    j_tpo = pd.DataFrame({"J_TPO_Velocity": ext_map_buffer})

    return j_tpo


def KVO(
    price_data: pd.DataFrame,
    fast_ema: int = 34,
    slow_ema: int = 55,
    signal_ema: int = 13,
) -> pd.DataFrame:
    """
    Calculates the Klinger Volume Oscillator (KVO) indicator.

    Parameters:
    price_data (pd.DataFrame): DataFrame of OHLCV data.
    FastEMA (int): The number of periods for the fast EMA. Default is 34.
    SlowEMA (int): The number of periods for the slow EMA. Default is 55.
    SignalEMA (int): The number of periods for the signal EMA. Default is 13.

    Returns:
    pd.DataFrame: A DataFrame of the KVO indicator values.
    """
    tpc = (price_data["High"] + price_data["Low"] + price_data["Close"]) / 3
    tpp = tpc.shift(-1)
    v = np.where(
        tpc > tpp, price_data["Volume"], np.where(tpc < tpp, -price_data["Volume"], 0)
    )

    MainBuffer = (
        pd.Series(v).ewm(span=fast_ema).mean() - pd.Series(v).ewm(span=slow_ema).mean()
    )
    SignalBuffer = MainBuffer.ewm(span=signal_ema).mean()

    kvo = pd.DataFrame({"KVO": MainBuffer, "KVO_Signal": SignalBuffer})

    return kvo


def LWPI(price_data: pd.DataFrame, period: int = 8) -> pd.DataFrame:
    """
    Larry Williams Proxy Index indicator

    Args:
    price_data: Pandas DataFrame of OHLCV data
    length: Length of the indicator period

    Returns:
    Pandas DataFrame of Larry Williams Proxy Index values
    """
    Raw = price_data["Open"] - price_data["Close"]
    MA = Raw.rolling(window=period).mean()
    ATR = (price_data["High"] - price_data["Low"]).rolling(window=period).mean()
    LWPI = 50 * MA / ATR + 50
    LWPI[ATR == 0] = 0
    return pd.DataFrame({"LWPI": LWPI})


def SuperTrend(
    price_data: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculate the Supertrend indicator from OHLCV price data.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame containing OHLCV price data.
    nbr_periods (int, optional): Number of periods for ATR calculation. Default is 10.
    multiplier (float, optional): Multiplier for ATR. Default is 3.0.

    Returns:
    pd.DataFrame: DataFrame containing the Supertrend indicator values.
    """
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]

    median_price = (high + low) / 2
    atr_values = ATR(price_data, period)
    up = median_price + multiplier * atr_values
    down = median_price - multiplier * atr_values
    trend = pd.Series(1, index=price_data.index)

    change_of_trend = 0
    for i in range(1, len(price_data)):
        if close[i] > up[i - 1]:
            trend[i] = 1
            if trend[i - 1] == -1:
                change_of_trend = 1
        elif close[i] < down[i - 1]:
            trend[i] = -1
            if trend[i - 1] == 1:
                change_of_trend = 1
        else:
            trend[i] = trend[i - 1]
            change_of_trend = 0

        flag = 1 if trend[i] < 0 and trend[i - 1] > 0 else 0
        flagh = 1 if trend[i] > 0 and trend[i - 1] < 0 else 0

        if trend[i] > 0 and down[i] < down[i - 1]:
            down[i] = down[i - 1]
        if trend[i] < 0 and up[i] > up[i - 1]:
            up[i] = up[i - 1]

        if flag == 1:
            up[i] = median_price[i] + multiplier * atr_values[i]
        if flagh == 1:
            down[i] = median_price[i] - multiplier * atr_values[i]

        if trend[i] == 1:
            supertrend = down[i]
        elif change_of_trend == 1:
            supertrend = up[i + 1]
            change_of_trend = 0
        elif trend[i] == -1:
            supertrend = up[i]
        elif change_of_trend == 1:
            supertrend = down[i + 1]
            change_of_trend = 0

    supertrend = pd.DataFrame({"Supertrend": supertrend, "Trend": trend})

    return supertrend


def TTF(
    price_data: pd.DataFrame,
    period: int = 8,
    top_line: int = 75,
    bottom_line: int = -75,
    t3_period: int = 3,
    b: float = 0.7,
) -> pd.DataFrame:
    """
    Trend Trigger Factor (TTF) indicator.

    Args:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    period (int): Number of bars for computation.
    top_line (int): Top line value.
    bottom_lin (int): Bottom line value.
    t3_period (int): Period of T3.
    b (float): Value of b.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the calculated indicator's values.
    """
    highest_high_recent = price_data["High"].rolling(window=period).max()
    highest_high_older = highest_high_recent.shift(period)
    lowest_low_recent = price_data["Low"].rolling(window=period).min()
    lowest_low_older = lowest_low_recent.shift(period)

    buy_power = highest_high_recent - lowest_low_older
    sell_power = highest_high_older - lowest_low_recent
    ttf = (buy_power - sell_power) / (0.5 * (buy_power + sell_power)) * 100

    c1 = -(b**3)
    c2 = 3 * b**2 + 3 * b**3
    c3 = -6 * b**2 - 3 * b - 3 * b**3
    c4 = 1 + 3 * b + b**2 + 3 * b**3
    t3 = (
        c1 * ttf
        + c2 * ttf.shift(t3_period)
        + c3 * ttf.shift(2 * t3_period)
        + c4 * ttf.shift(3 * t3_period)
    ).rename("T3")

    e1 = t3
    e2 = e1.ewm(span=2, adjust=False).mean()
    e3 = e2.ewm(span=2, adjust=False).mean()
    e4 = e3.ewm(span=2, adjust=False).mean()
    e5 = e4.ewm(span=2, adjust=False).mean()
    e6 = e5.ewm(span=2, adjust=False).mean()
    ttf = (c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3).rename("TTF")

    signal = np.where(ttf >= 0, top_line, bottom_line)

    return pd.DataFrame({"TTF": ttf, "Signal": signal}, index=price_data.index)


def Vortex(price_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates the Vortex Indicator (VI) and Vortex Movement (VM) for a given length.

    Parameters:
    price_data (pd.DataFrame): OHLCV data
    VI_Length (int): length of VI calculation (default=14)

    Returns:
    pd.DataFrame: VI+ and VI- values
    """
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]
    tr = pd.Series(data=0.0, index=price_data.index)
    plus_vm = abs(high - low.shift())
    minus_vm = abs(low - high.shift())

    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

    sum_plus_vm = plus_vm.rolling(window=period).sum()
    sum_minus_vm = minus_vm.rolling(window=period).sum()
    sum_tr = tr.rolling(window=period).sum()

    plus_vi = sum_plus_vm / sum_tr
    minus_vi = sum_minus_vm / sum_tr

    return pd.DataFrame({"PlusVI": plus_vi, "MinusVI": minus_vi})


def BraidFilterHist(
    price_data: pd.DataFrame,
    ma1_period: int = 3,
    ma2_period: int = 7,
    ma3_period: int = 14,
    atr_period: int = 14,
    pips_min_sep_percent: float = 40,
) -> pd.DataFrame:
    """
    Braid Filter indicator of Robert Hill stocks and commodities magazine 2006.

    Args:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    ma_period1 (int): Period of first moving average.
    ma_period2 (int): Period of second moving average.
    ma_period3 (int): Period of third moving average.
    atr_period (int): Period of ATR.
    pips_min_sep_percent (float): Minimum separation percent; separates MAs by minimum this % of ATR.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the calculated indicator's values.
    """
    atr = ATR(price_data, atr_period)
    ma1 = price_data["Close"].rolling(window=ma1_period).mean()
    ma2 = price_data["Open"].rolling(window=ma2_period).mean()
    ma3 = price_data["Close"].rolling(window=ma3_period).mean()

    max_val = pd.concat([ma1, ma2, ma3], axis=1).max(axis=1)
    min_val = pd.concat([ma1, ma2, ma3], axis=1).min(axis=1)
    diff = max_val - min_val
    fil = atr * pips_min_sep_percent / 100

    ma1_gt_ma2 = ma1 > ma2
    ma2_gt_ma1 = ma2 > ma1
    diff_gt_fil = diff > fil

    trend = pd.Series(
        np.where(
            ma1_gt_ma2 & diff_gt_fil, 1, np.where(ma2_gt_ma1 & diff_gt_fil, -1, 0)
        ),
        index=price_data.index,
    )
    trend.fillna(method="ffill", inplace=True)

    UpH = pd.Series(np.where(trend == 1, diff, np.nan), index=price_data.index)
    DnH = pd.Series(np.where(trend == -1, diff, np.nan), index=price_data.index)

    return pd.DataFrame({"UpH": UpH, "DnH": DnH}, index=price_data.index)


def BraidFilter(
    price_data: pd.DataFrame,
    period1: int = 5,
    period2: int = 8,
    period3: int = 20,
    pips_min_sep_percent: float = 0.5,
) -> pd.DataFrame:
    """
    Braid Filter indicator of Robert Hill stocks and commodities magazine 2006.

    Args:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    period1 (int): Period of first moving average.
    period2 (int): Period of second moving average.
    period3 (int): Period of third moving average.
    pips_min_sep_percent (int): Minimum separation percent; separates MAs by minimum this % of ATR.
        Specifies the minimum separation between the three moving averages as a fraction of the current average true range (ATR) value.
        The ATR is a measure of volatility that takes into account the range of price movement of an asset over a certain number of periods. By multiplying the ATR with pips_min_sep_percent/100, we get the minimum separation between the moving averages in pips.
        For example, if pips_min_sep_percent is set to 40, and the current ATR value is 100 pips, the minimum separation between the three moving averages would be 40% of 100 pips, or 40 pips.
        This parameter is used to filter out noise and prevent the three moving averages from getting too close to each other, which could result in false signals.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the calculated indicator's values.
    """
    ema1 = price_data["Close"].ewm(span=period1, adjust=False).mean()
    ema2 = price_data["Open"].ewm(span=period2, adjust=False).mean()
    ema3 = price_data["Close"].ewm(span=period3, adjust=False).mean()

    CrossUp, CrossDown = np.where((ema1 > ema2) & (ema2 > ema3), 1, 0), np.where(
        (ema1 < ema2) & (ema2 < ema3), -1, 0
    )
    ATR = price_data["High"] - price_data["Low"]
    Filter = ATR.rolling(window=14).mean().abs() * pips_min_sep_percent

    # Return a DataFrame
    return pd.DataFrame(
        {"CrossUp": CrossUp, "CrossDown": CrossDown, "Filter": Filter},
        index=price_data.index,
    )


def Laguerre(price_data: pd.DataFrame, gamma: float = 0.7) -> pd.DataFrame:
    """Calculates the Laguerre indicator.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame of OHLCV price data.
        gamma (float, optional): The gamma. Defaults to 0.7.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the calculated indicator's values.
    """
    close = price_data["Close"].values

    laguerre = np.zeros(len(price_data))

    i = len(price_data) - 1
    l0, l1, l2, l3, lrsi = 0, 0, 0, 0, 0

    while i >= 0:
        l0a, l1a, l2a, l3a = l0, l1, l2, l3

        l0 = (1 - gamma) * close[i] + gamma * l0a
        l1 = -gamma * l0 + l0a + gamma * l1a
        l2 = -gamma * l1 + l1a + gamma * l2a
        l3 = -gamma * l2 + l2a + gamma * l3a

        cu, cd = 0, 0

        if l0 >= l1:
            cu = l0 - l1
        else:
            cd = l1 - l0

        if l1 >= l2:
            cu += l1 - l2
        else:
            cd += l2 - l1

        if l2 >= l3:
            cu += l2 - l3
        else:
            cd += l3 - l2

        if cu + cd != 0:
            lrsi = cu / (cu + cd)

        laguerre[i] = lrsi

        i -= 1

    df = pd.DataFrame({"Laguerre": laguerre}, index=price_data.index)

    return df


def RecursiveMA(price_data: pd.DataFrame, period=2, recursions=20):
    data_length = len(price_data)
    open_prices = price_data["Open"].values

    xema_buffer = np.zeros(data_length)
    trigger_buffer = np.zeros(data_length)

    for i in range(data_length - 1, -1, -1):
        ema = np.full(recursions, open_prices[i])
        alpha = 2.0 / (period + 1.0)

        for j in range(recursions - 1):
            ema[1:] = alpha * ema[:-1] + (1 - alpha) * ema[1:]

        xema_buffer[i] = ema[-1]
        trigger_buffer[i] = (
            np.dot(ema, np.arange(recursions, 0, -1))
            * 2
            / (recursions * (recursions + 1))
        )

    recursive_ma = pd.DataFrame(
        {"Xema": xema_buffer, "Trigger": trigger_buffer}, index=price_data.index
    )
    return recursive_ma


def SchaffTrendCycle(
    price_data: pd.DataFrame,
    period: int = 10,
    fast_ma_period: int = 23,
    slow_ma_period: int = 50,
    signal_period: int = 3,
) -> pd.DataFrame:
    """
    Calculates the Schaff Trend Cycle (STC) indicator.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame containing OHLCV price data.
    period (int, optional): Schaff period. Default is 10.
    fast_ma_period (int, optional): Fast MACD period. Default is 23.
    slow_ma_period (int, optional): Slow MACD period. Default is 50.
    signal_period (int, optional): Signal period. Default is 3.

    Returns:
    pd.DataFrame: Pandas DataFrame containing the STC values.
    """

    # Calculate MACD
    macd = EMA(price_data, fast_ma_period) - EMA(price_data, slow_ma_period)

    # Calculate fastK and fastD
    macd_low = macd.rolling(window=period).min()
    macd_high = macd.rolling(window=period).max()
    fast_k = 100 * (macd - macd_low) / (macd_high - macd_low)
    alpha = 2.0 / (1.0 + signal_period)
    fast_d = fast_k.ewm(alpha=alpha, adjust=False).mean()

    # Calculate STC values
    stoch_low = fast_d.rolling(window=period).min()
    stoch_high = fast_d.rolling(window=period).max()
    fast_kk = 100 * (fast_d - stoch_low) / (stoch_high - stoch_low)
    stc = fast_kk.ewm(alpha=alpha, adjust=False).mean()

    # Calculate STC difference between current and previous values
    diff = stc.diff().abs()

    # Create DataFrame with STC values
    stc_df = pd.DataFrame({"STC": stc, "Diff": diff}, index=price_data.index)

    return stc_df


def SmoothStep(
    price_data: pd.DataFrame,
    period: int = 32,
) -> pd.DataFrame:
    """
    Calculates the SmoothStep indicator.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame containing OHLCV price data.
    period (int, optional): Period. Default is 32.

    Returns:
    pd.DataFrame: Pandas DataFrame containing the SmoothStep values.
    """
    price = price_data["Close"]

    # Calculate SmoothStep values
    min_price = price.rolling(window=period).min()
    max_price = price.rolling(window=period).max()
    raw_value = (price - min_price) / (max_price - min_price)
    smooth_step = raw_value**2 * (3 - 2 * raw_value)

    # Create DataFrame with SmoothStep values
    smooth_step_df = pd.DataFrame({"SmoothStep": smooth_step}, index=price_data.index)

    return smooth_step_df


def TopTrend(
    price_data: pd.DataFrame,
    period: int = 20,
    deviation: int = 2,
    money_risk: float = 1.00,
) -> pd.DataFrame:
    """
    Calculates the TopTrend indicator.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame containing OHLCV price data.
    period (int, optional): Bollinger Bands Period. Default is 20.
    deviation (int, optional): Deviation. Default is 2.
    money_risk (float, optional): Offset Factor. Default is 1.00.

    Returns:
    pd.DataFrame: Pandas DataFrame containing the TopTrend indicator values.
    """
    close = price_data["Close"]
    data_length = len(close)

    # Calculate Bollinger Bands
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    smax = sma + deviation * std
    smin = sma - deviation * std

    # Initialize arrays
    trend = np.zeros(data_length)
    bsmax = np.zeros(data_length)
    bsmin = np.zeros(data_length)
    uptrend_buffer = np.full(data_length, np.nan)
    downtrend_buffer = np.full(data_length, np.nan)

    for shift in range(data_length - 1, -1, -1):
        if shift < len(close) - 1:
            if close.iloc[shift] > smax.iloc[shift + 1]:
                trend[shift] = 1
            if close.iloc[shift] < smin.iloc[shift + 1]:
                trend[shift] = -1

            if trend[shift] > 0 and smin.iloc[shift] < smin.iloc[shift + 1]:
                smin.iloc[shift] = smin.iloc[shift + 1]
            if trend[shift] < 0 and smax.iloc[shift] > smax.iloc[shift + 1]:
                smax.iloc[shift] = smax.iloc[shift + 1]

            bsmax[shift] = smax.iloc[shift] + 0.5 * (money_risk - 1) * (
                smax.iloc[shift] - smin.iloc[shift]
            )
            bsmin[shift] = smin.iloc[shift] - 0.5 * (money_risk - 1) * (
                smax.iloc[shift] - smin.iloc[shift]
            )

            if trend[shift] > 0 and bsmin[shift] < bsmin[shift + 1]:
                bsmin[shift] = bsmin[shift + 1]
            if trend[shift] < 0 and bsmax[shift] > bsmax[shift + 1]:
                bsmax[shift] = bsmax[shift + 1]

            if trend[shift] > 0:
                uptrend_buffer[shift] = bsmin[shift]
                downtrend_buffer[shift] = np.nan
            elif trend[shift] < 0:
                downtrend_buffer[shift] = bsmax[shift]
                uptrend_buffer[shift] = np.nan

    # For the final data point
    if close.iloc[-1] > smax.iloc[-1]:
        trend[-1] = 1
    elif close.iloc[-1] < smin.iloc[-1]:
        trend[-1] = -1
    else:
        trend[-1] = trend[-2]

    # Create the result DataFrame
    result = pd.DataFrame({"TopTrend": trend}, index=price_data.index)

    return result


def TrendLord(
    price_data: pd.DataFrame,
    period: int = 12,
    ma_method: str = "smma",
    applied_price: str = "close",
    show_high_low: bool = False,
    signal_bar: int = 1,
) -> pd.DataFrame:
    """
    Calculate the TrendLord indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        Pandas DataFrame containing OHLCV price data.
    period : int, default 12
        The number of periods for the moving average.
    ma_method : str, default 'smma'
        The moving average method: 'sma', 'ema', 'smma', or 'lwma'.
    applied_price : str, default 'close'
        The price to apply the moving average to: 'open', 'high', 'low', 'close', 'hl2', 'hlc3', or 'ohlc4'.
    show_high_low : bool, default False
        Whether to use the high/low values in the calculation.
    signal_bar : int, default 1
        The bar to signal the indicator value.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the calculated indicator values.
    """
    # Map the 'applied_price' string to the corresponding price values
    if applied_price == "open":
        price = price_data["Open"]
    elif applied_price == "high":
        price = price_data["High"]
    elif applied_price == "low":
        price = price_data["Low"]
    elif applied_price == "close":
        price = price_data["Close"]
    elif applied_price == "hl2":
        price = (price_data["High"] + price_data["Low"]) / 2
    elif applied_price == "hlc3":
        price = (price_data["High"] + price_data["Low"] + price_data["Close"]) / 3
    elif applied_price == "ohlc4":
        price = (
            price_data["Open"]
            + price_data["High"]
            + price_data["Low"]
            + price_data["Close"]
        ) / 4

    # Calculate the moving average based on the 'ma_method' parameter
    if ma_method == "sma":
        MA = price.rolling(window=period).mean()
    elif ma_method == "ema":
        MA = price.ewm(span=period).mean()
    elif ma_method == "smma":
        MA = price.ewm(alpha=1 / period).mean()
    elif ma_method == "lwma":
        weights = np.arange(1, period + 1)
        MA = price.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    # Calculate Array1 as the moving average of MA
    if ma_method == "sma":
        Array1 = MA.rolling(window=period).mean()
    elif ma_method == "ema":
        Array1 = MA.ewm(span=period).mean()
    elif ma_method == "smma":
        Array1 = MA.ewm(alpha=1 / period).mean()
    elif ma_method == "lwma":
        weights = np.arange(1, period + 1)
        Array1 = MA.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    # Initialize the MAIN and SIGNAL arrays (SELL and BUYY in og MQL4)
    MAIN = np.zeros_like(price)
    SIGNAL = np.zeros_like(price)

    # Iterate over the data and calculate the BUYY and SELL values
    for i in range(signal_bar, len(price)):
        slotLL = price_data["Low"][i] if show_high_low else MA[i]
        slotHH = price_data["High"][i] if show_high_low else MA[i]

        if Array1[i] > Array1[i - 1]:
            SIGNAL[i] = slotLL
            MAIN[i] = Array1[i]
        if Array1[i] < Array1[i - 1]:
            SIGNAL[i] = slotHH
            MAIN[i] = Array1[i]

    # Create a DataFrame to store the calculated indicator values
    indicator_values = pd.DataFrame({"Main": MAIN, "Signal": SIGNAL})

    return indicator_values


def TwiggsMF(price_data: pd.DataFrame, period: int = 21) -> pd.DataFrame:
    """
    Calculate the Twigg's Money Flow indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        Pandas DataFrame containing OHLCV price data.
    period : int, default 21
        The number of periods for the moving average.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the calculated indicator values.
    """
    # Check if there are at least 4 bars
    if len(price_data) < period:
        return pd.DataFrame()

    # Calculate TR, ADV, and Vol
    price_data["TRH"] = price_data[["High", "Close"]].max(axis=1)
    price_data["TRL"] = price_data[["Low", "Close"]].min(axis=1)
    price_data["TR"] = price_data["TRH"] - price_data["TRL"]

    price_data["ADV"] = (
        (2 * price_data["Close"] - price_data["TRL"] - price_data["TRH"])
        / price_data["TR"]
        * price_data["Volume"]
    )
    price_data["Vol"] = price_data["Volume"]

    # Handle any division by zero that may have occurred
    price_data["ADV"].replace([np.inf, -np.inf], 0, inplace=True)

    # Calculate k
    k = 2 / (period + 1)

    # Calculate WMA_ADV and WMA_V
    price_data["WMA_ADV"] = price_data["ADV"].ewm(span=period, adjust=False).mean()
    price_data["WMA_V"] = price_data["Vol"].ewm(span=period, adjust=False).mean()

    # Calculate TMF
    price_data["TMF"] = price_data["WMA_ADV"] / price_data["WMA_V"]

    # Create a DataFrame to store the calculated indicator values
    indicator_values = pd.DataFrame({"TMF": price_data["TMF"]})

    return indicator_values


def UF2018(price_data: pd.DataFrame, period: int = 54) -> pd.DataFrame:
    """
    Calculate the uf2018 indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        Pandas DataFrame containing OHLCV price data.
    period : int, default 54
        The period for the Zig Zag calculation.
    bar_n : int, default 1000
        The number of bars to calculate the indicator for.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the calculated indicator values.
    """
    SELL = pd.Series(index=price_data.index, dtype=float)
    BUY = pd.Series(index=price_data.index, dtype=float)

    li_20 = 0
    li_16 = 0
    index_24 = 0
    bar_n = len(price_data) - 1
    high_60 = price_data["High"][bar_n]
    low_68 = price_data["Low"][bar_n]
    down = False
    up = False

    for i in range(bar_n, -1, -1):
        low = 10000000
        high = -100000000

        for j in range(i + period, i, -1):
            if j > bar_n:
                continue
            if price_data["Low"][j] < low:
                low = price_data["Low"][j]
            if price_data["High"][j] > high:
                high = price_data["High"][j]

        if price_data["Low"][i] < low and price_data["High"][i] > high:
            li_16 = 2
        else:
            if price_data["Low"][i] < low:
                li_16 = -1
            if price_data["High"][i] > high:
                li_16 = 1

        if li_16 != li_20 and li_20 != 0:
            if li_16 == 2:
                li_16 = -li_20
                high_60 = price_data["High"][i]
                low_68 = price_data["Low"][i]
                down = False
                up = False

            index_24 += 1

            up = True if li_16 == 1 else False
            down = True if li_16 == -1 else False

            high_60 = price_data["High"][i]
            low_68 = price_data["Low"][i]

        if li_16 == 1 and price_data["High"][i] >= high_60:
            high_60 = price_data["High"][i]

        if li_16 == -1 and price_data["Low"][i] <= low_68:
            low_68 = price_data["Low"][i]

        li_20 = li_16

        BUY[i] = 1 if up else 0
        SELL[i] = 1 if down else 0

    # Return the results as a DataFrame
    return pd.DataFrame({"BUY": BUY, "SELL": SELL})


def LSMA(df: pd.DataFrame, period: int = 14, shift: int = 0) -> pd.Series:
    """
    Least Squares Moving Average (LSMA) calculation.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' column.
    period (int): Period for LSMA calculation.
    shift (int): Number of periods to shift the source data.

    Returns:
    pd.Series: Series of LSMA values.
    """
    lengthvar = (period + 1) / 3
    weights = np.array([(i - lengthvar) for i in range(period, 0, -1)])

    # Define a rolling apply function using dot product of weights
    def calc_lsma(window):
        return np.dot(weights, window) * 6 / (period * (period + 1))

    shifted_close = df["Close"].shift(shift)
    lsma_series = shifted_close.rolling(window=period).apply(calc_lsma, raw=True)

    return lsma_series


def AcceleratorLSMA(
    df: pd.DataFrame, long_period: int = 21, short_period: int = 9
) -> pd.DataFrame:
    """
    Calculate Accelerator/Decelerator Oscillator

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' column

    Returns:
    pd.DataFrame: DataFrame with calculated values
    """

    limit = len(df)
    ExtBuffer3 = np.array(
        [LSMA(df, short_period, i) - LSMA(df, long_period, i) for i in range(limit)]
    )
    ExtBuffer4 = pd.Series(ExtBuffer3).rolling(window=short_period).mean().values

    current = ExtBuffer3 - ExtBuffer4
    prev = np.roll(current, 1)
    up = current > prev

    ExtBuffer1 = np.where(up, current, 0)
    ExtBuffer2 = np.where(~up, current, 0)
    ExtBuffer0 = current

    result = pd.DataFrame(
        {
            "ExtBuffer0": ExtBuffer0,
            "ExtBuffer1": ExtBuffer1,
            "ExtBuffer2": ExtBuffer2,
            "ExtBuffer3": ExtBuffer3,
            "ExtBuffer4": ExtBuffer4,
        }
    )

    return result


def SSL(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    Calculate SSL channels.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close', 'High', and 'Low' columns
    lb (int): Lookback period

    Returns:
    pd.DataFrame: DataFrame with calculated values
    """
    # Calculate rolling high and low
    high_sma = df["High"].rolling(window=period).mean()
    low_sma = df["Low"].rolling(window=period).mean()

    # Initialize hlv as a DataFrame of zeros
    hlv = pd.Series(0, index=df.index)

    # Conditions for hlv
    hlv[df["Close"] > high_sma] = 1
    hlv[df["Close"] < low_sma] = -1
    hlv = hlv.ffill()

    # Initialize ssld and sslu
    ssld = pd.Series(index=df.index, dtype=float)
    sslu = pd.Series(index=df.index, dtype=float)

    # Conditions for ssld and sslu
    ssld[hlv == 1] = low_sma
    ssld[hlv == -1] = high_sma
    sslu[hlv == 1] = high_sma
    sslu[hlv == -1] = low_sma

    ssld = ssld.ffill()
    sslu = sslu.ffill()

    return pd.DataFrame({"SSL_Down": ssld, "SSL_Up": sslu})


# Volume Functions


def ADX(price_data: pd.DataFrame, period=14) -> pd.DataFrame:
    # Calculate the True Range (TR)
    tr = np.maximum(
        price_data["High"] - price_data["Low"],
        np.maximum(
            abs(price_data["High"] - price_data["Close"].shift()),
            abs(price_data["Low"] - price_data["Close"].shift()),
        ),
    )

    # Calculate the Directional Movement (DM) and True Directional Movement (TDM)
    dm_plus = np.where(
        price_data["High"] - price_data["High"].shift()
        > price_data["Low"].shift() - price_data["Low"],
        price_data["High"] - price_data["High"].shift(),
        0,
    )
    dm_minus = np.where(
        price_data["Low"].shift() - price_data["Low"]
        > price_data["High"] - price_data["High"].shift(),
        price_data["Low"].shift() - price_data["Low"],
        0,
    )
    tdm_plus = pd.Series(dm_plus).rolling(period).sum()
    tdm_minus = pd.Series(dm_minus).rolling(period).sum()

    # Calculate the Positive Directional Index (+DI) and Negative Directional Index (-DI)
    di_plus = 100 * tdm_plus / tr.rolling(period).sum()
    di_minus = 100 * tdm_minus / tr.rolling(period).sum()

    # Calculate the Directional Movement Index (DX)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

    # Calculate the ADX
    adx = dx.rolling(period).mean()

    result = pd.DataFrame(
        {
            "ADX": adx,
            "+DI": di_plus,
            "-DI": di_minus,
        }
    )

    # Return the dataframe with the ADX values
    return result


def TDFI(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """
    Calculate the Trend Direction Force Index (TDFI) using a Pandas DataFrame.

    :param df: Pandas DataFrame containing OHLC and Volume columns
    :param period: Period for the Exponential Moving Average (EMA) calculation, default is 13
    :return: Pandas Series containing the TDFI values
    """
    # Calculate the Force Index
    force_index = (df["Close"] - df["Close"].shift(1)) * df["Volume"]

    # Calculate the smoothing factor
    smoothing_factor = 2 / (period + 1)

    # Calculate the smoothed Force Index (TDFI) using EMA
    smoothed_force_index = force_index.ewm(alpha=smoothing_factor, adjust=False).mean()

    # Normalize the smoothed Force Index to range between -1 and 1
    tdfi = (
        smoothed_force_index / smoothed_force_index.abs().rolling(window=period).max()
    )

    # Convert to DataFrame
    tdfi = pd.DataFrame({"TDFI": tdfi})

    return tdfi


def WAE(
    price_data: pd.DataFrame,
    minutes: int = 0,
    sensitivity: int = 150,
    dead_zone_pip: int = 15,
) -> pd.DataFrame:
    """
    Calculate the Waddah Attar Explosion indicator values and return them as a Pandas DataFrame.

    Parameters:
    price_data (pd.DataFrame): Input DataFrame containing OHLCV columns.
    minutes (int): Timeframe in minutes. Default is 0.
    sensitivity (int): Sensitivity parameter. Default is 150.
    dead_zone_pip (int): Dead zone pip parameter. Default is 15.

    Returns:
    pd.DataFrame: DataFrame of calculated indicator values.
    """
    # Resample price data to the desired timeframe
    if minutes > 0:
        price_data = price_data.resample(f"{minutes}T").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )

    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def macd(
        series: pd.Series, fast_period: int, slow_period: int, signal_period: int
    ) -> pd.Series:
        fast_ema = ema(series, fast_period)
        slow_ema = ema(series, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, signal_period)
        return macd_line, signal_line

    def bollinger_bands(series: pd.Series, period: int, std_dev: float) -> pd.DataFrame:
        middle = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper = middle + (std_dev * std_dev)
        lower = middle - (std_dev * std_dev)
        return upper, middle, lower

    close = price_data["Close"]

    # Calculate MACD
    macd_line, signal_line = macd(
        close, fast_period=20, slow_period=40, signal_period=9
    )

    # Calculate Bollinger Bands
    upper, middle, lower = bollinger_bands(close, period=20, std_dev=2)

    # Calculate Trend and Explo
    trend = (macd_line - signal_line) * sensitivity
    explo = upper - lower

    # Calculate Dead zone
    dead_zone = price_data["Close"].apply(lambda x: x * dead_zone_pip)

    # Create output DataFrame
    output = pd.DataFrame(index=price_data.index)
    output["Trend"] = trend
    output["Explosion"] = explo
    output["Dead"] = dead_zone

    return output


def NormalizedVolume(price_data, period: int = 14) -> pd.DataFrame:
    """Calculates the Normalized Volume indicator values for a given OHLCV dataframe.

    Args:
        price_data (_type_): A Pandas DataFrame of OHLCV price data.
        period (int, optional): The period for the MA of the volume. Defaults to 14.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the normalized volume values.
    """
    volume = price_data["Volume"].values

    volume_buffer = volume[::-1].copy()

    # Calculate MA of volume buffer
    volume_ma = pd.Series(volume_buffer).rolling(period).mean().values

    vol = volume / volume_ma * 100
    up = np.where(vol > 100, vol, np.nan)
    dn = np.where(vol <= 100, vol, np.nan)

    return pd.DataFrame({"Vol": vol}, index=price_data.index)


def VolatilityRatio(
    price_data: pd.DataFrame, period: int = 25, inp_price: str = "Close"
) -> pd.DataFrame:
    """
    Calculate the Volatility Ratio.

    Parameters:
    price_data (pd.DataFrame): OHLCV price data
    period (int): Volatility period
    inp_price (str): Column name in the price_data DataFrame for the price (default 'Close')

    Returns:
    pd.DataFrame: DataFrame containing the Volatility Ratio values
    """
    rates_total = len(price_data)
    price = price_data[inp_price].values

    val = np.empty(rates_total, dtype=float)
    valda = np.empty(rates_total, dtype=float)
    valdb = np.empty(rates_total, dtype=float)
    valc = np.empty(rates_total, dtype=int)

    m_array = {
        "price": np.zeros(rates_total),
        "price2": np.zeros(rates_total),
        "sum": np.zeros(rates_total),
        "sum2": np.zeros(rates_total),
        "sumd": np.zeros(rates_total),
        "deviation": np.zeros(rates_total),
    }

    for i in range(rates_total - 1, -1, -1):
        m_array["price"][i] = price[i]
        m_array["price2"][i] = price[i] * price[i]

        if i > period:
            m_array["sum"][i] = (
                m_array["sum"][i - 1]
                + m_array["price"][i]
                - m_array["price"][i - period]
            )
            m_array["sum2"][i] = (
                m_array["sum2"][i - 1]
                + m_array["price2"][i]
                - m_array["price2"][i - period]
            )
        else:
            m_array["sum"][i] = m_array["price"][i]
            m_array["sum2"][i] = m_array["price2"][i]
            for k in range(1, period):
                if i >= k:
                    m_array["sum"][i] += m_array["price"][i - k]
                    m_array["sum2"][i] += m_array["price2"][i - k]

        m_array["deviation"][i] = np.sqrt(
            (m_array["sum2"][i] - m_array["sum"][i] * m_array["sum"][i] / period)
            / period
        )

        if i > period:
            m_array["sumd"][i] = (
                m_array["sumd"][i - 1]
                + m_array["deviation"][i]
                - m_array["deviation"][i - period]
            )
        else:
            m_array["sumd"][i] = m_array["deviation"][i]
            for k in range(1, period):
                if i >= k:
                    m_array["sumd"][i] += m_array["deviation"][i - k]

        deviation_average = m_array["sumd"][i] / period
        val[i] = (
            m_array["deviation"][i] / deviation_average if deviation_average != 0 else 1
        )
        valc[i] = 1 if val[i] > 1 else 2 if val[i] < 1 else 0

        if valc[i] == 2:
            valda[i] = valdb[i] = np.nan
        elif valc[i] == 1:
            valda[i] = val[i]
            valdb[i] = np.nan
        else:
            valda[i] = np.nan
            valdb[i] = val[i]

    vr = pd.DataFrame(
        {"VR": val, "VR Up": valda, "VR Down": valdb}, index=price_data.index
    )

    return vr


# TODO
def TCF(
    price_data: pd.DataFrame,
    period: int = 20,
    count_bars: int = 5000,
    t3_period: int = 5,
    b: float = 0.618,
) -> pd.DataFrame:

    close = price_data["Close"]
    bars = len(close)

    accounted_bars = max(0, bars - count_bars)

    t3_k_p = pd.Series(index=close.index, dtype=float)
    t3_k_n = pd.Series(index=close.index, dtype=float)

    for cnt in range(accounted_bars, bars):
        shift = bars - 1 - cnt

        change_p = close.shift(-1) - close
        change_n = close - close.shift(-1)

        cf_p = change_p.where(change_p > 0, 0).cumsum()
        cf_n = change_n.where(change_n > 0, 0).cumsum()

        ch_p = change_p[shift : shift + period].sum()
        ch_n = change_n[shift : shift + period].sum()
        cff_p = cf_p[shift : shift + period].sum()
        cff_n = cf_n[shift : shift + period].sum()

        k_p = ch_p - cff_n
        k_n = ch_n - cff_p

        a1 = k_p
        a2 = k_n

        e1 = e2 = e3 = e4 = e5 = e6 = 0
        e12 = e22 = e32 = e42 = e52 = e62 = 0

        b2 = b * b
        b3 = b2 * b
        c1 = -b3
        c2 = 3 * (b2 + b3)
        c3 = -3 * (2 * b2 + b + b3)
        c4 = 1 + 3 * b + b3 + 3 * b2

        n1 = t3_period
        if n1 < 1:
            n1 = 1
        n1 = 1 + 0.5 * (n1 - 1)
        w1 = 2 / (n1 + 1)
        w2 = 1 - w1

        for _ in range(t3_period):
            e1 = w1 * a1 + w2 * e1
            e2 = w1 * e1 + w2 * e2
            e3 = w1 * e2 + w2 * e3
            e4 = w1 * e3 + w2 * e4
            e5 = w1 * e4 + w2 * e5
            e6 = w1 * e5 + w2 * e6

            e12 = w1 * a2 + w2 * e12
            e22 = w1 * e12 + w2 * e22
            e32 = w1 * e22 + w2 * e32
            e42 = w1 * e32 + w2 * e42
            e52 = w1 * e42 + w2 * e52
            e62 = w1 * e52 + w2 * e62

        t3_k_p[shift] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        t3_k_n[shift] = c1 * e62 + c2 * e52 + c3 * e42 + c4 * e32

    tcf = pd.DataFrame({"T3KP": t3_k_p, "T3KN": t3_k_n})

    return tcf


# TODO
def DSP(data, signal_period=9, dsp_period=14):
    pass
