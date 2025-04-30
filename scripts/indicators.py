import inspect
import logging
import multiprocessing

import pandas as pd
import talib as ta
from tqdm import tqdm

from scripts import config
from scripts.indicator_functions import (
    ALMA,
    EHLERS,
    HMA,
    J_TPO,
    KASE,
    KVO,
    LSMA,
    LWPI,
    SSL,
    TCF,
    TDFI,
    TTF,
    UF2018,
    VIDYA,
    WAE,
    AcceleratorLSMA,
    BraidFilter,
    BraidFilterHist,
    BullsBearsImpulse,
    CenterOfGravity,
    Coral,
    FantailVMA,
    FilteredATR,
    Fisher,
    Gen3MA,
    GruchaIndex,
    HalfTrend,
    KalmanFilter,
    KijunSen,
    Laguerre,
    MACDZeroLag,
    McGinleyDI,
    NormalizedVolume,
    RecursiveMA,
    SchaffTrendCycle,
    SmoothStep,
    SuperTrend,
    TopTrend,
    TrendLord,
    TwiggsMF,
    VolatilityRatio,
    Vortex,
)
from scripts.signal_functions import *
from scripts.sql import HistoricalDataSQLHelper

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)


# =============================================================================
# TA‑lib–Based Indicators
# =============================================================================

ta_lib_candlestick = [
    {
        "name": "CANDLE_2CROWS",
        "function": lambda df, **kwargs: ta.CDL2CROWS(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Two Crows: A bearish reversal pattern where two consecutive bearish candles signal a potential trend change from bullish to bearish.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3BLACKCROWS",
        "function": lambda df, **kwargs: ta.CDL3BLACKCROWS(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Three Black Crows: A strong bearish reversal pattern characterized by three long bearish candles with short shadows, suggesting heavy selling.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3INSIDE",
        "function": lambda df, **kwargs: ta.CDL3INSIDE(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Three Inside: A pattern where a small candle is engulfed by its neighbors, hinting at potential reversal or pause in the current trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3LINESTRIKE",
        "function": lambda df, **kwargs: ta.CDL3LINESTRIKE(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Three Line Strike: A reversal pattern where three candles are followed by a candle that ‘strikes’ through them, indicating a possible trend reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3STARSINSOUTH",
        "function": lambda df, **kwargs: ta.CDL3STARSINSOUTH(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Three Stars in the South: A bullish reversal pattern with three small candles after a downtrend, suggesting emerging buying pressure.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3WHITESOLDIERS",
        "function": lambda df, **kwargs: ta.CDL3WHITESOLDIERS(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Three White Soldiers: A bullish reversal pattern marked by three consecutive strong white candles, indicating sustained upward momentum.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_ABANDONEDBABY",
        "function": lambda df, **kwargs: ta.CDLABANDONEDBABY(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Abandoned Baby: A rare and powerful reversal pattern featuring an isolated doji with gaps on both sides, signaling a sharp change in trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_ADVANCEBLOCK",
        "function": lambda df, **kwargs: ta.CDLADVANCEBLOCK(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Advance Block: A bearish reversal pattern that appears after an uptrend when bullish candles become constrained, hinting at exhaustion.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_BELTHOLD",
        "function": lambda df, **kwargs: ta.CDLBELTHOLD(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Belt Hold: A reversal signal where a long candle opens at an extreme (low for bullish, high for bearish) and moves strongly, indicating momentum.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_BREAKAWAY",
        "function": lambda df, **kwargs: ta.CDLBREAKAWAY(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Breakaway: A pattern that forms after a gap and a series of candles, signaling the end of a trend and the potential start of a reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_CLOSINGMARUBOZU",
        "function": lambda df, **kwargs: ta.CDLCLOSINGMARUBOZU(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Closing Marubozu: A candlestick with little or no shadows that shows strong momentum in the direction of the close.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_CONCEALBABYSWALL",
        "function": lambda df, **kwargs: ta.CDLCONCEALBABYSWALL(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Concealing Baby Swallow: A bearish reversal pattern where a small candle is engulfed, suggesting a shift in market sentiment.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_COUNTERATTACK",
        "function": lambda df, **kwargs: ta.CDLCOUNTERATTACK(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Counterattack: A pattern where a gap in one direction is met by a candle in the opposite direction, implying a potential reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_DARKCLOUDCOVER",
        "function": lambda df, **kwargs: ta.CDLDARKCLOUDCOVER(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Dark Cloud Cover: A bearish reversal pattern where a bearish candle opens above and closes below the midpoint of a preceding bullish candle.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_DOJI",
        "function": lambda df, **kwargs: ta.CDLDOJI(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Doji: Represents indecision in the market as the open and close are nearly equal; it may signal a reversal when seen with other indicators.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_DOJISTAR",
        "function": lambda df, **kwargs: ta.CDLDOJISTAR(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Doji Star: A variation of the doji that appears with a gap, highlighting indecision and a possible imminent reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_DRAGONFLYDOJI",
        "function": lambda df, **kwargs: ta.CDLDRAGONFLYDOJI(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Dragonfly Doji: A bullish reversal signal with a long lower shadow and no upper shadow, indicating that buyers may step in.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_ENGULFING",
        "function": lambda df, **kwargs: ta.CDLENGULFING(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Engulfing: A reversal pattern where one candle completely engulfs the previous candle’s body, suggesting a strong change in sentiment.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_EVENINGDOJISTAR",
        "function": lambda df, **kwargs: ta.CDLEVENINGDOJISTAR(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Evening Doji Star: A bearish reversal pattern where a doji follows a bullish candle, signaling potential weakness.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_EVENINGSTAR",
        "function": lambda df, **kwargs: ta.CDLEVENINGSTAR(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Evening Star: A classic bearish reversal pattern formed by three candles, marking a potential shift from uptrend to downtrend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_GAPSIDESIDEWHITE",
        "function": lambda df, **kwargs: ta.CDLGAPSIDESIDEWHITE(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Gap Side-by-Side White Lines: Typically a continuation pattern where similar white candles with gaps indicate trend persistence.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_GRAVESTONEDOJI",
        "function": lambda df, **kwargs: ta.CDLGRAVESTONEDOJI(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Gravestone Doji: A bearish reversal signal with a long upper shadow and no lower shadow, suggesting a potential market top.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HAMMER",
        "function": lambda df, **kwargs: ta.CDLHAMMER(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Hammer: A bullish reversal pattern in a downtrend, characterized by a small body and a long lower shadow indicating buying support.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HANGINGMAN",
        "function": lambda df, **kwargs: ta.CDLHANGINGMAN(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Hanging Man: A bearish reversal pattern in an uptrend with a small body and long lower shadow, hinting at potential weakness.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HARAMI",
        "function": lambda df, **kwargs: ta.CDLHARAMI(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Harami: A reversal pattern where a small candle is contained within the previous candle’s body, suggesting a possible trend change.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HARAMICROSS",
        "function": lambda df, **kwargs: ta.CDLHARAMICROSS(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Harami Cross: Similar to the Harami pattern but with a doji, indicating uncertainty and a potential reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HIGHWAVE",
        "function": lambda df, **kwargs: ta.CDLHIGHWAVE(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "High-Wave Candle: Features long shadows with a small body, reflecting high volatility and market indecision.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HIKKAKE",
        "function": lambda df, **kwargs: ta.CDLHIKKAKE(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Hikkake: A trap pattern that identifies false breakouts and signals a potential reversal when the breakout fails.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HIKKAKEMOD",
        "function": lambda df, **kwargs: ta.CDLHIKKAKEMOD(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Modified Hikkake: A refined version of the Hikkake pattern that offers a clearer reversal signal after a false breakout.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HOMINGPIGEON",
        "function": lambda df, **kwargs: ta.CDLHOMINGPIGEON(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Homing Pigeon: A bullish reversal pattern where a small bullish candle appears in a downtrend, suggesting a potential bottom.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_IDENTICAL3CROWS",
        "function": lambda df, **kwargs: ta.CDLIDENTICAL3CROWS(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Identical Three Crows: A bearish reversal pattern with three similarly shaped bearish candles, indicating strong selling pressure.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_INNECK",
        "function": lambda df, **kwargs: ta.CDLINNECK(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "In-Neck: A bearish reversal pattern where the close is near the previous candle’s low, hinting at continuing downward momentum.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_INVERTEDHAMMER",
        "function": lambda df, **kwargs: ta.CDLINVERTEDHAMMER(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Inverted Hammer: A bullish reversal pattern after a downtrend, marked by a small body with a long upper shadow that signals buying interest.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_KICKING",
        "function": lambda df, **kwargs: ta.CDLKICKING(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Kicking: A reversal pattern marked by a gap and a strong candle in the opposite direction, indicating a sharp change in market sentiment.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_KICKINGBYLENGTH",
        "function": lambda df, **kwargs: ta.CDLKICKINGBYLENGTH(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Kicking by Length: A variant of the Kicking pattern that emphasizes longer candles, suggesting a more powerful reversal signal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_LADDERBOTTOM",
        "function": lambda df, **kwargs: ta.CDLLADDERBOTTOM(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Ladder Bottom: A bullish reversal pattern that forms near support, indicating the potential end of a downtrend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_LONGLEGGEDDOJI",
        "function": lambda df, **kwargs: ta.CDLLONGLEGGEDDOJI(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Long-Legged Doji: Exhibits a wide range with a nearly equal open and close, signifying indecision that may lead to a reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_LONGLINE",
        "function": lambda df, **kwargs: ta.CDLLONGLINE(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Long Line Candle: A candle with a long body, indicating strong momentum in the prevailing trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MARUBOZU",
        "function": lambda df, **kwargs: ta.CDLMARUBOZU(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Marubozu: A candlestick with no shadows that reflects dominance by buyers or sellers, supporting trend continuation.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MATCHINGLOW",
        "function": lambda df, **kwargs: ta.CDLMATCHINGLOW(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Matching Low: A bullish reversal pattern where the current low matches the previous low, hinting at a price bottom.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MATHOLD",
        "function": lambda df, **kwargs: ta.CDLMATHOLD(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Mat Hold: A bullish continuation pattern that shows a pause before an uptrend resumes, suggesting temporary consolidation.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MORNINGDOJISTAR",
        "function": lambda df, **kwargs: ta.CDLMORNINGDOJISTAR(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Morning Doji Star: A bullish reversal pattern where a doji appears between a bearish and a bullish candle, indicating a potential bottom.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MORNINGSTAR",
        "function": lambda df, **kwargs: ta.CDLMORNINGSTAR(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Morning Star: A well-known bullish reversal pattern formed by three candles that signals the end of a downtrend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_ONNECK",
        "function": lambda df, **kwargs: ta.CDLONNECK(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "On-Neck: A bearish reversal pattern where the second candle closes near the first candle’s neck, suggesting emerging selling pressure.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_PIERCING",
        "function": lambda df, **kwargs: ta.CDLPIERCING(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Piercing: A bullish reversal pattern where a bearish candle is followed by a bullish candle that closes above the midpoint of the previous candle.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_RICKSHAWMAN",
        "function": lambda df, **kwargs: ta.CDLRICKSHAWMAN(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Rickshaw Man: A pattern similar to a doji that reflects indecision, potentially signaling an upcoming reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_RISEFALL3METHODS",
        "function": lambda df, **kwargs: ta.CDLRISEFALL3METHODS(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Rising/Falling Three Methods: A continuation pattern where a group of candles indicates that the current trend is still intact despite minor pullbacks.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_SEPARATINGLINES",
        "function": lambda df, **kwargs: ta.CDLSEPARATINGLINES(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Separating Lines: A continuation pattern where similar candles appear consecutively, suggesting the trend will persist.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_SHOOTINGSTAR",
        "function": lambda df, **kwargs: ta.CDLSHOOTINGSTAR(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Shooting Star: A bearish reversal pattern in an uptrend, marked by a small body and long upper shadow, indicating potential trend weakness.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_SHORTLINE",
        "function": lambda df, **kwargs: ta.CDLSHORTLINE(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Short Line Candle: A candle with a narrow range that reflects low volatility and market indecision.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_SPINNINGTOP",
        "function": lambda df, **kwargs: ta.CDLSPINNINGTOP(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Spinning Top: Characterized by a small body with long shadows, this pattern signals indecision and may precede a reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_STALLEDPATTERN",
        "function": lambda df, **kwargs: ta.CDLSTALLEDPATTERN(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Stalled Pattern: A reversal signal in an uptrend that indicates a loss of momentum and a potential market top.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_STICKSANDWICH",
        "function": lambda df, **kwargs: ta.CDLSTICKSANDWICH(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Stick Sandwich: A continuation pattern where a small candle is ‘sandwiched’ between two similar candles, suggesting the trend will continue.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_TAKURI",
        "function": lambda df, **kwargs: ta.CDLTAKURI(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Takuri: A pattern akin to a dragonfly doji with a long lower shadow, indicating a potential reversal when buyers gain control.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_TASUKIGAP",
        "function": lambda df, **kwargs: ta.CDLTASUKIGAP(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Tasuki Gap: A continuation pattern where a gap between candles is partially filled, supporting the ongoing trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_THRUSTING",
        "function": lambda df, **kwargs: ta.CDLTHRUSTING(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Thrusting: A bearish reversal pattern where a candle opens within the previous candle’s body and closes near its low, indicating selling pressure.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_TRISTAR",
        "function": lambda df, **kwargs: ta.CDLTRISTAR(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Tristar: A reversal pattern composed of three small candles that signal indecision and a potential change in trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_UNIQUE3RIVER",
        "function": lambda df, **kwargs: ta.CDLUNIQUE3RIVER(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Unique 3 River: A less common reversal pattern indicating a marked shift in momentum over three consecutive candles.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_UPSIDEGAP2CROWS",
        "function": lambda df, **kwargs: ta.CDLUPSIDEGAP2CROWS(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Upside Gap Two Crows: A bearish reversal pattern where an initial gap up is followed by two bearish candles, suggesting a trend reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_XSIDEGAP3METHODS",
        "function": lambda df, **kwargs: ta.CDLXSIDEGAP3METHODS(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "X-Side Gap Three Methods: A continuation pattern identified by a gap and three supportive candles, indicating the current trend is likely to persist.",
        "signal_function": interpret_candlestick,
    },
]

# =============================================================================
# Volume Indicators
# =============================================================================

ta_lib_volume = [
    {
        "name": "AD",
        "function": lambda df, **kwargs: ta.AD(
            df["High"], df["Low"], df["Close"], df["Volume"], **kwargs
        ),
        "description": "Accumulation/Distribution (AD): Measures the cumulative flow of money into and out of a security based on price and volume.",
        "signal_function": signal_ad,
        "raw_function": ta.AD,
        "parameters": {},
    },
    {
        "name": "ADX",
        "function": lambda df, **kwargs: ta.ADX(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "Average Directional Index (ADX): Measures the strength of a trend but not its direction.",
        "signal_function": signal_adx,
        "raw_function": ta.ADX,
        "parameters": {"timeperiod": 14},
    },
    {
        "name": "ADXR",
        "function": lambda df, **kwargs: ta.ADXR(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "Average Directional Movement Rating (ADXR): Smoothed version of ADX to confirm trend strength.",
        "signal_function": signal_adx,
        "raw_function": ta.ADXR,
        "parameters": {"timeperiod": 14},
    },
    {
        "name": "ADOSC",
        "function": lambda df, **kwargs: ta.ADOSC(
            df["High"], df["Low"], df["Close"], df["Volume"], **kwargs
        ),
        "description": "Accumulation/Distribution Oscillator (ADOSC): A momentum oscillator based on the AD line to detect volume strength shifts.",
        "signal_function": signal_adosc,
        "raw_function": ta.ADOSC,
        "parameters": {"fastperiod": 3, "slowperiod": 10},
    },
    {
        "name": "OBV",
        "function": lambda df, **kwargs: ta.OBV(df["Close"], df["Volume"], **kwargs),
        "description": "On-Balance Volume (OBV): Tracks cumulative buying/selling pressure by adding volume on up days and subtracting it on down days.",
        "signal_function": signal_obv,
        "raw_function": ta.OBV,
        "parameters": {},
    },
]
additional_volume = [
    {
        "name": "FantailVMA",
        "function": lambda df, **kwargs: FantailVMA(df, **kwargs),
        "description": "Fantail VMA: A smoothed volume-weighted average that responds to price volatility and ADX to infer market momentum.",
        "signal_function": signal_fantail_vma,
        "raw_function": FantailVMA,
        "parameters": {"adx_length": 2, "weighting": 2.0, "ma_length": 1},
    },
    {
        "name": "KVO",
        "function": lambda df, **kwargs: KVO(df, **kwargs),
        "description": "Klinger Volume Oscillator (KVO): Combines price movements and volume to identify long-term trends of money flow.",
        "signal_function": signal_kvo,
        "raw_function": KVO,
        "parameters": {"fast_ema": 34, "slow_ema": 55, "signal_ema": 13},
    },
    {
        "name": "LWPI",
        "function": lambda df, **kwargs: LWPI(df, **kwargs),
        "description": "Larry Williams Proxy Index (LWPI): An oscillator evaluating buying/selling strength using price and volatility.",
        "signal_function": signal_lwpi,
        "raw_function": LWPI,
        "parameters": {"period": 8},
    },
    {
        "name": "NormalizedVolume",
        "function": lambda df, **kwargs: NormalizedVolume(df, **kwargs),
        "description": "Normalized Volume: Expresses current volume as a percentage of the average over a given period.",
        "signal_function": signal_normalized_volume,
        "raw_function": NormalizedVolume,
        "parameters": {"period": 14},
    },
    {
        "name": "TwiggsMF",
        "function": lambda df, **kwargs: TwiggsMF(df, **kwargs),
        "description": "Twiggs Money Flow: Combines price and volume to estimate the strength of accumulation or distribution.",
        "signal_function": signal_twiggs_mf,
        "raw_function": TwiggsMF,
        "parameters": {"period": 21},
    },
]
volume_indicators = ta_lib_volume + additional_volume

# =============================================================================
# Trend Indicators
# =============================================================================

ta_lib_trend = [
    {
        "name": "AROON",
        "function": lambda df, **kwargs: ta.AROON(df["High"], df["Low"], **kwargs),
        "description": "Aroon Down: Indicates how recently the lowest low occurred during a given period.",
        "signal_function": signal_aroon,
        "raw_function": ta.AROON,
        "parameters": {"timeperiod": 14},
    },
    {
        "name": "AROONOSC",
        "function": lambda df, **kwargs: ta.AROONOSC(df["High"], df["Low"], **kwargs),
        "description": "Aroon Oscillator: Measures the difference between Aroon Up and Down to gauge trend direction.",
        "signal_function": signal_aroonosc,
        "raw_function": ta.AROONOSC,
        "parameters": {"timeperiod": 14},
    },
]
additional_trend = [
    {
        "name": "KASE",
        "function": lambda df, **kwargs: KASE(df, **kwargs),
        "description": "KASE Permission Stochastic: Smoothed stochastic oscillator used to assess trade permission based on trend momentum.",
        "signal_function": signal_kase,
        "raw_function": KASE,
        "parameters": {"pstLength": 9, "pstX": 5, "pstSmooth": 3, "smoothPeriod": 10},
    },
    {
        "name": "KijunSen",
        "function": lambda df, **kwargs: KijunSen(df, **kwargs),
        "description": "Kijun-sen (Base Line): Part of Ichimoku, provides key support/resistance levels and trend direction.",
        "signal_function": signal_kijunsen,
        "raw_function": KijunSen,
        "parameters": {"period": 26, "shift": 9},
    },
    {
        "name": "KalmanFilter",
        "function": lambda df, **kwargs: KalmanFilter(df, **kwargs),
        "description": "Kalman Filter: A statistical filter used to smooth price data and reveal trend dynamics.",
        "signal_function": signal_kalman_filter,
        "raw_function": KalmanFilter,
        "parameters": {"k": 1, "sharpness": 1},
    },
    {
        "name": "SSL",
        "function": lambda df, **kwargs: SSL(df, **kwargs),
        "description": "SSL Channel: A trend-following indicator that uses smoothed high/low averages to generate crossovers.",
        "signal_function": signal_ssl,
        "raw_function": SSL,
        "parameters": {"period": 10},
    },
    {
        "name": "SuperTrend",
        "function": lambda df, **kwargs: SuperTrend(df, **kwargs),
        "description": "SuperTrend: A trailing stop and trend direction indicator based on ATR.",
        "signal_function": signal_supertrend,
        "raw_function": SuperTrend,
        "parameters": {"period": 10, "multiplier": 3.0},
    },
    {
        "name": "TrendLord",
        "function": lambda df, **kwargs: TrendLord(df, **kwargs),
        "description": "TrendLord: A smoothed oscillator-based trend detector showing bullish/bearish strength zones.",
        "signal_function": signal_trendlord,
        "raw_function": TrendLord,
        "parameters": {
            "period": 12,
            "ma_method": "smma",
            "applied_price": "close",
            "show_high_low": False,
            "signal_bar": 1,
        },
    },
    {
        "name": "UF2018",
        "function": lambda df, **kwargs: UF2018(df, **kwargs),
        "description": "UF2018: A visual zig-zag style trend-following indicator that shows direction based on local highs/lows.",
        "signal_function": signal_uf2018,
        "raw_function": UF2018,
        "parameters": {"period": 54},
    },
    {
        "name": "CenterOfGravity",
        "function": lambda df, **kwargs: CenterOfGravity(df, **kwargs),
        "description": "Center of Gravity (COG): A zero-lag oscillator that forecasts turning points using weighted averages.",
        "signal_function": signal_center_of_gravity,
        "raw_function": CenterOfGravity,
        "parameters": {"period": 10},
    },
    {
        "name": "GruchaIndex",
        "function": lambda df, **kwargs: GruchaIndex(df, **kwargs),
        "description": "Grucha Index: A visual trend indicator showing buyer vs. seller dominance over recent candles.",
        "signal_function": signal_grucha_index,
        "raw_function": GruchaIndex,
        "parameters": {"period": 10, "ma_period": 10},
    },
    {
        "name": "HalfTrend",
        "function": lambda df, **kwargs: HalfTrend(df, **kwargs),
        "description": "HalfTrend: A non-lagging trend reversal indicator that reacts only after sustained movement.",
        "signal_function": signal_halftrend,
        "raw_function": HalfTrend,
        "parameters": {"amplitude": 2},
    },
    {
        "name": "TopTrend",
        "function": lambda df, **kwargs: TopTrend(df, **kwargs),
        "description": "TopTrend: Uses modified Bollinger logic to indicate trend reversals and trailing stop levels.",
        "signal_function": signal_top_trend,
        "raw_function": TopTrend,
        "parameters": {"period": 20, "deviation": 2, "money_risk": 1.0},
    },
]
trend_indicators = ta_lib_trend + additional_trend

# =============================================================================
# Momentum Indicators
# =============================================================================

ta_lib_momentum = [
    {
        "name": "APO",
        "function": lambda df, fastperiod=12, slowperiod=26: ta.APO(
            df["Close"], fastperiod=fastperiod, slowperiod=slowperiod
        ),
        "description": "APO (Absolute Price Oscillator): Measures the difference between fast and slow EMAs of price to indicate momentum direction and strength.",
        "signal_function": signal_apo,
        "raw_function": ta.APO,
        "parameters": {
            "fastperiod": 12,
            "slowperiod": 26,
            "matype": 0,
        },  # matype goes 0-8
    },
    {
        "name": "CCI",
        "function": lambda df, **kwargs: ta.CCI(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "CCI (Commodity Channel Index): A momentum oscillator that measures deviation from a moving average to detect overbought or oversold conditions.",
        "signal_function": signal_cci,
        "raw_function": ta.CCI,
        "parameters": {"timeperiod": 14},
    },
    {
        "name": "MACD",
        "function": lambda df, **kwargs: ta.MACD(df["Close"], **kwargs),
        "description": "MACD: Combines the MACD line, signal line, and histogram to track momentum and detect entry signals via crossovers.",
        "signal_function": signal_macd,
        "raw_function": ta.MACD,
        "parameters": {
            "fastperiod": 12,
            "slowperiod": 26,
            "signalperiod": 9,
        },
    },
    {
        "name": "MACDEXT",
        "function": lambda df, **kwargs: ta.MACDEXT(df["Close"], **kwargs),
        "description": "MACDEXT: An extended version of the MACD indicator with customizable moving average types.",
        "signal_function": signal_macdext,
        "raw_function": ta.MACDEXT,
        "parameters": {
            "fastperiod": 12,
            "fastmatype": 0,  # matype goes 0-8
            "slowperiod": 26,
            "slowmatype": 0,  # matype goes 0-8
            "signalperiod": 9,
            "signalmatype": 0,  # matype goes 0-8
        },
    },
    {
        "name": "MACDFIX",
        "function": lambda df, **kwargs: ta.MACDFIX(df["Close"], **kwargs),
        "description": "MACDFIX: A simplified MACD that uses a fixed signal period (typically 9), useful for consistent momentum detection.",
        "signal_function": signal_macdfix,
        "raw_function": ta.MACDFIX,
        "parameters": {"signalperiod": 9},
    },
    {
        "name": "MAMA",
        "function": lambda df, **kwargs: ta.MAMA(df["Close"], **kwargs),
        "description": "MAMA: The MESA Adaptive Moving Average is designed to react faster to price changes using adaptive cycle techniques.",
        "signal_function": signal_mama,
        "raw_function": ta.MAMA,
        "parameters": {"fastlimit": 0, "slowlimit": 0},
    },
    {
        "name": "PPO",
        "function": lambda df, **kwargs: ta.PPO(df["Close"], **kwargs),
        "description": "PPO (Percentage Price Oscillator): A normalized MACD that measures momentum as a percentage rather than absolute price.",
        "signal_function": signal_ppo,
        "raw_function": ta.PPO,
        "parameters": {
            "fastperiod": 12,
            "slowperiod": 26,
            "matype": 0,
        },  # matype goes 0-8
    },
    {
        "name": "ROC",
        "function": lambda df, **kwargs: ta.ROC(df["Close"], **kwargs),
        "description": "ROC (Rate of Change): Measures the percentage change in price over a defined period to assess momentum shifts.",
        "signal_function": signal_roc,
        "raw_function": ta.ROC,
        "parameters": {"timeperiod": 10},
    },
    {
        "name": "ROCP",
        "function": lambda df, **kwargs: ta.ROCP(df["Close"], **kwargs),
        "description": "ROCP (Rate of Change Percent): Measures the percent change in price over a specified period to detect momentum shifts.",
        "signal_function": signal_rocp,
        "raw_function": ta.ROCP,
        "parameters": {"timeperiod": 10},
    },
    {
        "name": "ROCR",
        "function": lambda df, **kwargs: ta.ROCR(df["Close"], **kwargs),
        "description": "ROCR (Rate of Change Ratio): Expresses price change as a ratio relative to a previous period, highlighting acceleration or deceleration.",
        "signal_function": signal_rocr,
        "raw_function": ta.ROCR,
        "parameters": {"timeperiod": 10},
    },
    {
        "name": "ROCR100",
        "function": lambda df, **kwargs: ta.ROCR100(df["Close"], **kwargs),
        "description": "ROCR100: Like ROCR, but scaled to 100. A value >100 indicates upward momentum; <100 suggests decline.",
        "signal_function": signal_rocr100,
        "raw_function": ta.ROCR100,
        "parameters": {"timeperiod": 10},
    },
    {
        "name": "RSI",
        "function": lambda df, **kwargs: ta.RSI(df["Close"], **kwargs),
        "description": "RSI (Relative Strength Index): Measures recent price gains versus losses to identify overbought or oversold conditions.",
        "signal_function": signal_rsi,
        "raw_function": ta.RSI,
        "parameters": {"timeperiod": 14},
    },
    {
        "name": "STOCH_Slow",
        "function": lambda df, **kwargs: ta.STOCH(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "Stochastic Oscillator (Slow): Uses smoothed %K and %D crossovers to identify potential momentum reversals.",
        "signal_function": signal_stoch,
        "raw_function": ta.STOCH,
        "parameters": {
            "fastk_period": 5,
            "slowk_period": 3,
            "slowk_matype": 0,  # matype goes 0-8
            "slowd_period": 3,
            "slowd_matype": 0,  # matype goes 0-8
        },
    },
    {
        "name": "STOCHF_Fast",
        "function": lambda df, **kwargs: ta.STOCHF(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "Stochastic Oscillator (Fast): Uses raw %K and %D crossovers for faster, more sensitive momentum shifts.",
        "signal_function": signal_stoch,
        "raw_function": ta.STOCHF,
        "parameters": {
            "fastk_period": 5,
            "fastd_period": 3,
            "fastd_matype": 0,
        },  # matype goes 0-8
    },
    {
        "name": "STOCHRSI",
        "function": lambda df, **kwargs: ta.STOCHRSI(df["Close"], **kwargs),
        "description": "Stochastic RSI: Applies stochastic logic to RSI values, enhancing detection of overbought/oversold extremes.",
        "signal_function": signal_stoch,
        "raw_function": ta.STOCHRSI,
        "parameters": {
            "timeperiod": 14,
            "fastk_period": 5,
            "fastd_period": 3,
            "fastd_matype": 0,
        },
    },
    {
        "name": "TSF",
        "function": lambda df, **kwargs: ta.TSF(df["Close"], **kwargs),
        "description": "TSF (Time Series Forecast): Projects a linear regression forward, estimating future price direction.",
        "signal_function": signal_tsf,
        "raw_function": ta.TSF,
        "parameters": {"timeperiod": 14},
    },
    {
        "name": "ULTOSC",
        "function": lambda df, **kwargs: ta.ULTOSC(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "ULTOSC (Ultimate Oscillator): Combines multiple timeframes of momentum into one oscillator to reduce false signals.",
        "signal_function": signal_ultosc,
        "raw_function": ta.ULTOSC,
        "parameters": {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28},
    },
]
additional_momentum = [
    {
        "name": "MACDZeroLag",
        "function": lambda df, **kwargs: MACDZeroLag(df, **kwargs),
        "description": "MACDZeroLag: A variation of MACD designed to reduce lag by applying zero-lag moving averages for better signal timing.",
        "signal_function": signal_macd_zero_lag,
        "raw_function": MACDZeroLag,
        "parameters": {"short_period": 12, "long_period": 26, "signal_period": 9},
    },
    {
        "name": "Fisher",
        "function": lambda df, **kwargs: Fisher(df, **kwargs),
        "description": "Fisher Transform: Sharpens turning points in price action using a mathematical transformation of normalized prices.",
        "signal_function": signal_fisher,
        "raw_function": Fisher,
        "parameters": {
            "range_periods": 10,
            "price_smoothing": 0.3,
            "index_smoothing": 0.3,
        },
    },
    {
        "name": "BullsBearsImpulse",
        "function": lambda df, **kwargs: BullsBearsImpulse(df, **kwargs),
        "description": "Bulls/Bears Impulse: Measures the relative strength of bullish and bearish forces to highlight dominant sentiment.",
        "signal_function": signal_bulls_bears_impulse,
        "raw_function": BullsBearsImpulse,
        "parameters": {"ma_period": 13},
    },
    {
        "name": "J_TPO",
        "function": lambda df, **kwargs: J_TPO(df, **kwargs),
        "description": "J_TPO: A custom oscillator derived from time-price opportunity modeling to reflect short-term velocity and acceleration.",
        "signal_function": signal_j_tpo,
        "raw_function": J_TPO,
        "parameters": {"period": 14},
    },
    {
        "name": "Laguerre",
        "function": lambda df, **kwargs: Laguerre(df, **kwargs),
        "description": "Laguerre Filter: A smooth oscillator designed to track price momentum while minimizing whipsaws.",
        "signal_function": signal_laguerre,
        "raw_function": Laguerre,
        "parameters": {"gamma": 0.7},
    },
    {
        "name": "SchaffTrendCycle",
        "function": lambda df, **kwargs: SchaffTrendCycle(df, **kwargs),
        "description": "Schaff Trend Cycle: Combines MACD and cycle theory to create a responsive momentum oscillator.",
        "signal_function": signal_schaff_trend_cycle,
        "raw_function": SchaffTrendCycle,
        "parameters": {
            "period": 10,
            "fast_ma_period": 23,
            "slow_ma_period": 50,
            "signal_period": 3,
        },
    },
    {
        "name": "TDFI",
        "function": lambda df, **kwargs: TDFI(df, **kwargs),
        "description": "TDFI (Trend Direction & Force Index): Captures the intensity and direction of price moves for momentum analysis.",
        "signal_function": signal_tdfi,
        "raw_function": TDFI,
        "parameters": {"period": 13},
    },
    {
        "name": "TTF",
        "function": lambda df, **kwargs: TTF(df, **kwargs),
        "description": "TTF (Trend Trigger Factor): A smoothed oscillator designed to confirm momentum reversals through threshold crossings.",
        "signal_function": signal_ttf,
        "raw_function": TTF,
        "parameters": {
            "period": 8,
            "top_line": 75,
            "bottom_line": -75,
            "t3_period": 3,
            "b": 0.7,
        },
    },
    {
        "name": "AcceleratorLSMA",
        "function": lambda df, **kwargs: AcceleratorLSMA(df, **kwargs),
        "description": "Accelerator LSMA: Combines LSMA smoothing with acceleration logic to detect momentum shifts early.",
        "signal_function": signal_accelerator_lsma,
        "raw_function": AcceleratorLSMA,
        "parameters": {"long_period": 30, "short_period": 10},
    },
]
momentum_indicators = ta_lib_momentum + additional_momentum

# =============================================================================
# Volatility Indicators
# =============================================================================

ta_lib_volatility = [
    {
        "name": "ATR",
        "function": lambda df, **kwargs: ta.ATR(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "Average True Range (ATR): Measures absolute volatility by accounting for gaps and high-low range.",
        "signal_function": signal_volatility_line,
        "raw_function": ta.ATR,
        "parameters": {"timeperiod": 14},
    },
    {
        "name": "NATR",
        "function": lambda df, **kwargs: ta.NATR(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "Normalized ATR (NATR): ATR expressed as a percentage of price, useful for cross-asset volatility comparison.",
        "signal_function": signal_volatility_line,
        "raw_function": ta.NATR,
        "parameters": {"timeperiod": 14},
    },
    {
        "name": "TRANGE",
        "function": lambda df, **kwargs: ta.TRANGE(
            df["High"], df["Low"], df["Close"], **kwargs
        ),
        "description": "True Range (TRANGE): Raw measure of price range and gap movement, used as a base for ATR.",
        "signal_function": signal_volatility_line,
        "raw_function": ta.TRANGE,
        "parameters": {},
    },
    {
        "name": "BollingerBands",
        "function": lambda df, **kwargs: ta.BBANDS(df["Close"], **kwargs),
        "description": "Bollinger Bands: A volatility-based envelope plotted at standard deviations above and below a moving average. Returns upper, middle, and lower bands.",
        "signal_function": signal_bollinger,
        "raw_function": ta.BBANDS,
        "parameters": {
            "timeperiod": 5,
            "nbdevup": 2,
            "nbdevdn": 2,
            "matype": 0,
        },
    },
    {
        "name": "STDDEV",
        "function": lambda df, **kwargs: ta.STDDEV(df["Close"], **kwargs),
        "description": "Standard Deviation (STDDEV): Measures price dispersion from the mean to gauge volatility.",
        "signal_function": signal_stddev,
        "raw_function": ta.STDDEV,
        "parameters": {"timeperiod": 5, "nbdev": 1},
    },
    {
        "name": "VAR",
        "function": lambda df, **kwargs: ta.VAR(df["Close"], **kwargs),
        "description": "Variance (VAR): Square of standard deviation; tracks volatility by measuring price fluctuation strength.",
        "signal_function": signal_var,
        "raw_function": ta.VAR,
        "parameters": {"timeperiod": 5, "nbdev": 1},
    },
]
additional_volatility = [
    {
        "name": "FilteredATR",
        "function": lambda df, **kwargs: FilteredATR(df, **kwargs),
        "description": "Filtered ATR: A smoothed version of the ATR to reduce noise and better reflect sustained volatility.",
        "signal_function": signal_filtered_atr,
        "raw_function": FilteredATR,
        "parameters": {"period": 34, "ma_period": 34, "ma_shift": 0},
    },
    {
        "name": "VolatilityRatio",
        "function": lambda df, **kwargs: VolatilityRatio(df, **kwargs),
        "description": "Volatility Ratio: Compares recent price deviation against a longer-period range to measure relative volatility shifts.",
        "signal_function": signal_volatility_ratio,
        "raw_function": VolatilityRatio,
        "parameters": {"period": 25, "inp_price": "Close"},
    },
    {
        "name": "WAE",
        "function": lambda df, **kwargs: WAE(df, **kwargs),
        "description": "Waddah Attar Explosion (WAE): Combines MACD-based momentum with Bollinger Band expansion to highlight explosive volatility phases.",
        "signal_function": signal_wae,
        "raw_function": WAE,
        "parameters": {"minutes": 0, "sensitivity": 150, "dead_zone_pip": 15},
    },
]
volatility_indicators = ta_lib_volatility + additional_volatility

# =============================================================================
# Price/Statistical Indicators
# =============================================================================

ta_lib_price = [
    {
        "name": "AVGPRICE",
        "function": lambda df, **kwargs: ta.AVGPRICE(
            df["Open"], df["High"], df["Low"], df["Close"]
        ),
        "description": "Average Price: The average of the open, high, low, and close prices. A smoothed central value.",
        "signal_function": signal_avgprice,
    },
    {
        "name": "MEDPRICE",
        "function": lambda df, **kwargs: ta.MEDPRICE(df["High"], df["Low"]),
        "description": "Median Price: The midpoint between the high and low price.",
        "signal_function": signal_medprice,
    },
    {
        "name": "MAX",
        "function": lambda df, timeperiod=14: ta.MAX(
            df["Close"], timeperiod=timeperiod
        ),
        "description": "Maximum Price: Highest close price over the given period.",
        "signal_function": signal_max,
    },
    {
        "name": "MAXINDEX",
        "function": lambda df, timeperiod=14: ta.MAXINDEX(
            df["Close"], timeperiod=timeperiod
        ),
        "description": "Max Index: Index of the maximum closing price over the time period.",
        "signal_function": signal_maxindex,
    },
    {
        "name": "MIDPOINT",
        "function": lambda df, timeperiod=14: ta.MIDPOINT(
            df["Close"], timeperiod=timeperiod
        ),
        "description": "Midpoint: Average of highest and lowest close price over a time period.",
        "signal_function": signal_midpoint,
    },
    {
        "name": "MIDPRICE",
        "function": lambda df, timeperiod=14: ta.MIDPRICE(
            df["High"], df["Low"], timeperiod=timeperiod
        ),
        "description": "Mid Price: Average of high and low prices over the specified period.",
        "signal_function": signal_midprice,
    },
    {
        "name": "MIN",
        "function": lambda df, timeperiod=14: ta.MIN(
            df["Close"], timeperiod=timeperiod
        ),
        "description": "Minimum Price: Lowest close price over the given period.",
        "signal_function": signal_min,
    },
    {
        "name": "MININDEX",
        "function": lambda df, timeperiod=14: ta.MININDEX(
            df["Close"], timeperiod=timeperiod
        ),
        "description": "Min Index: Index of the minimum close price over the time period.",
        "signal_function": signal_minindex,
    },
    {
        "name": "MINMAX",
        "function": lambda df, timeperiod=14: ta.MINMAX(
            df["Close"], timeperiod=timeperiod
        ),
        "description": "MINMAX: Returns the lowest and highest close price over a given period, useful for detecting range and breakouts.",
        "signal_function": signal_minmax,
    },
    {
        "name": "SUM",
        "function": lambda df, timeperiod=14: ta.SUM(
            df["Close"], timeperiod=timeperiod
        ),
        "description": "Sum: Total of close prices over the defined time period.",
        "signal_function": signal_sum,
    },
    {
        "name": "TYPPRICE",
        "function": lambda df, **kwargs: ta.TYPPRICE(
            df["High"], df["Low"], df["Close"]
        ),
        "description": "Typical Price: Average of high, low, and close. Reflects a representative transaction price.",
        "signal_function": signal_typprice,
    },
    {
        "name": "WCLPRICE",
        "function": lambda df, **kwargs: ta.WCLPRICE(
            df["High"], df["Low"], df["Close"]
        ),
        "description": "Weighted Close Price: Weighted average price placing more emphasis on the close.",
        "signal_function": signal_wclprice,
    },
]
price_indicators = ta_lib_price

# =============================================================================
# Baseline (Moving Average–Based) Indicators
# =============================================================================
# These indicators are primarily built on smoothing or adaptive moving averages.

baseline_indicators = [
    {
        "name": "ALMA",
        "function": lambda df, **kwargs: ALMA(df, **kwargs),
        "description": "ALMA (Arnaud Legoux Moving Average): A Gaussian-weighted moving average designed to reduce lag while smoothing price.",
        "signal_function": signal_baseline_standard,
        "raw_function": ALMA,
        "parameters": {"period": 9, "sigma": 6, "offset": 0.85},
    },
    {
        "name": "HMA",
        "function": lambda df, **kwargs: HMA(df, **kwargs),
        "description": "HMA (Hull Moving Average): Uses weighted moving averages to minimize lag and smooth price action responsively.",
        "signal_function": signal_baseline_standard,
        "raw_function": HMA,
        "parameters": {"period": 13},
    },
    {
        "name": "RecursiveMA",
        "function": lambda df, **kwargs: RecursiveMA(df, **kwargs),
        "description": "Recursive MA: Applies repeated exponential smoothing to create a stable baseline for trend analysis.",
        "signal_function": signal_recursive_ma,
        "raw_function": RecursiveMA,
        "parameters": {"period": 2, "recursions": 20},
    },
    {
        "name": "LSMA",
        "function": lambda df, **kwargs: LSMA(df, **kwargs),
        "description": "LSMA (Least Squares Moving Average): A regression-based average used to project smoothed directional bias.",
        "signal_function": signal_lsma,
        "raw_function": LSMA,
        "parameters": {"period": 14, "shift": 0},
    },
    {
        "name": "VIDYA",
        "function": lambda df, **kwargs: VIDYA(df, **kwargs),
        "description": "VIDYA (Variable Index Dynamic Average): Adaptive moving average that responds to volatility by adjusting smoothing.",
        "signal_function": signal_baseline_standard,
        "raw_function": VIDYA,
        "parameters": {"period": 9, "histper": 30},
    },
    {
        "name": "Gen3MA",
        "function": lambda df, **kwargs: Gen3MA(df, **kwargs),
        "description": "Gen3MA: A third-generation moving average that combines multi-scale smoothing and sampling.",
        "signal_function": signal_gen3ma,
        "raw_function": Gen3MA,
        "parameters": {"period": 220, "sampling_period": 50},
    },
    {
        "name": "TrendLord",
        "function": lambda df, **kwargs: TrendLord(df, **kwargs),
        "description": "TrendLord: A smoothed MA-based trend filter that dynamically reacts to shifts using layered moving average logic.",
        "signal_function": signal_trendlord,
        "raw_function": TrendLord,
        "parameters": {
            "period": 12,
            "ma_method": "smma",
            "applied_price": "close",
            "show_high_low": False,
            "signal_bar": 1,
        },
    },
    {
        "name": "BraidFilter",
        "function": lambda df, **kwargs: BraidFilter(df, **kwargs),
        "description": "Braid Filter: Combines multiple moving averages and separation conditions to confirm trend stability or congestion.",
        "signal_function": signal_braidfilter,
        "raw_function": BraidFilter,
        "parameters": {
            "period1": 5,
            "period2": 8,
            "period3": 20,
            "pips_min_sep_percent": 0.5,
        },
    },
    {
        "name": "AcceleratorLSMA",
        "function": lambda df, **kwargs: AcceleratorLSMA(df, **kwargs),
        "description": "Accelerator LSMA: Applies differential velocity of LSMA to identify acceleration or deceleration in trend bias.",
        "signal_function": signal_accelerator_lsma,
        "raw_function": AcceleratorLSMA,
        "parameters": {"long_period": 21, "short_period": 9},
    },
    {
        "name": "McGinleyDI",
        "function": lambda df, **kwargs: McGinleyDI(df, **kwargs),
        "description": "McGinley Dynamic Index: An adaptive moving average that self-adjusts for speed and volatility.",
        "signal_function": signal_baseline_standard,
        "raw_function": McGinleyDI,
        "parameters": {"period": 12, "mcg_constant": 5},
    },
]

# =============================================================================
# Merge All Categories into a Master List if needed:
# =============================================================================

all_indicators = (
    ta_lib_candlestick
    + volume_indicators
    + trend_indicators
    + momentum_indicators
    + volatility_indicators
    + price_indicators
    + baseline_indicators
)


def need_to_calculate_indicators(data: pd.DataFrame) -> bool:
    """
    Checks if we need to calculate indicators on this data by
    checking if the last row of the DataFrame has any missing indicator values.

    Parameters:
    - data: pd.DataFrame, the dataframe containing price and indicator data.

    Returns:
    - bool: True if any data is missing in the last row, False otherwise.
    """
    if data.empty:
        return True  # No data means it's incomplete

    last_row = data.iloc[-1]  # Get the last row

    # Check if any of the indicator columns are missing in the last row
    for column in config.FINAL_INDICATOR_COLUMNS:
        matching_columns = [c for c in data.columns if column in c]

        # If no matching columns are found, return True to calculate
        if len(matching_columns) == 0:
            return True
        for c in matching_columns:
            try:
                if pd.isna(last_row[c]):
                    return True
            except KeyError:
                return True

    return False


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the columns match standard OHLCV format.
    Converts timestamp to datetime and sets index.
    """
    data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "vwap": "VWAP",
            "transactions": "Transactions",
        },
        inplace=True,
    )

    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data.set_index("Timestamp", inplace=True)
    data.sort_index(inplace=True)


def calculate_indicators(data: pd.DataFrame, indicators: list[dict]) -> pd.DataFrame:
    """
    Calculates all indicators and appends them to the DataFrame.
    """
    # logging.info(f"Calculating {len(indicators)} indicators...")

    indicator_results = {}  # Dictionary to store results before merging

    for indicator in indicators:
        indicator_function = indicator["function"]

        result = indicator["function"](data)

        if isinstance(result, pd.Series):
            column_name = indicator["name"]
            indicator_results[column_name] = result
        elif isinstance(result, pd.DataFrame):
            # Handle multi-column indicators like MACD, Bollinger Bands
            for column in result.columns:
                column_name = f"{indicator['name']}_{column}"
                indicator_results[column_name] = result[column]
        elif isinstance(result, np.ndarray):
            column_name = indicator["name"]
            series = pd.Series(result)
            indicator_results[column_name] = series
        elif isinstance(result, tuple):  # Tuple of numpy arrays
            i = 0
            for item in result:
                column_name = f"{indicator['name']}_{i}"
                item_series = pd.Series(item)
                indicator_results[column_name] = item_series
                i += 1
        else:
            raise ValueError(f"Unsupported indicator return type: {type(result)}")

    # Merge all indicators at once using pd.concat to prevent fragmentation
    data = pd.concat([data, pd.DataFrame(indicator_results, index=data.index)], axis=1)

    return data


def process_database(filename: str) -> str:
    # logging.info(f"Calculating indicators for {filename}...")

    sql_helper = HistoricalDataSQLHelper(f"data/{filename}")
    db_tables = sql_helper.get_database_tables()

    for table in db_tables:
        existing_data = sql_helper.get_historical_data(table)

        if not need_to_calculate_indicators(existing_data):
            # logging.info(f"Data collection for {table} already completed. Skipping...")
            continue

        normalize_data(existing_data)
        calculated_data = calculate_indicators(existing_data, all_indicators)
        calculated_data.reset_index(inplace=True)
        sql_helper.insert_historical_data(calculated_data, table)

    sql_helper.close_connection()

    return filename  # Return something so tqdm knows it's done


def calculate_indicators_for_all_databases() -> None:
    files = HistoricalDataSQLHelper.get_all_data_files()
    processes = round(multiprocessing.cpu_count() / 4)
    processes = 1
    with multiprocessing.Pool(processes=processes) as pool:
        with tqdm(
            total=len(files), desc="Calculating indicators for databases"
        ) as pbar:
            for result in pool.imap_unordered(process_database, files):
                pbar.update()
                logging.info(f"Calculated indicators for {result}.")


def extract_indicator_params(
    func, ignore_args=("open", "high", "low", "close", "volume")
):
    sig = inspect.signature(func)
    params = {}

    for name, param in sig.parameters.items():
        name_lower = name.lower()
        if name_lower not in ignore_args:
            if param.default is not inspect.Parameter.empty:
                params[name] = param.default
            else:
                params[name] = None  # Required, but no default

    return params
