from scripts import strategies
from scripts.indicators.indicator_configs import find_indicator_config


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


def load_strategy_from_dict(d: dict) -> strategies.BaseStrategy:
    """
    Expected dict shape (current NNFX):
      {
        "strategy_name": "NNFX",            # optional; default "NNFX"
        "pair": "EURUSD",
        "timeframe": "4_hour",
        "parameters": {
           "ATR": <IndicatorConfig>,
           "Baseline": <IndicatorConfig>,
           "C1": <IndicatorConfig>,
           "C2": <IndicatorConfig>,
           "Volume": <IndicatorConfig>,
           "Exit": <IndicatorConfig>,
        }
      }
    """
    name = d.get("strategy_name", "NNFX")
    pair = d["pair"]
    timeframe = d["timeframe"]

    params = d.get("parameters")
    if params is None:
        # Reconstruct params dict from your current row shape if needed
        # (Your existing code that maps raw fields to IndicatorConfig goes here)
        raise ValueError("Expected 'parameters' dict in strategy row.")

    return strategies.create_strategy(name, params, pair, timeframe)
