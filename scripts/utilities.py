from scripts import strategies
from scripts.indicators import find_indicator_config


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


def load_strategy_from_dict(strategy_json: dict) -> strategies.NNFXStrategy:
    ind = strategy_json["indicators"]

    atr = find_indicator_config(ind["ATR"]["name"], ind["ATR"]["parameters"])
    baseline = find_indicator_config(
        ind["BASELINE"]["name"], ind["BASELINE"]["parameters"]
    )
    c1 = find_indicator_config(ind["C1"]["name"], ind["C1"]["parameters"])
    c2 = find_indicator_config(ind["C2"]["name"], ind["C2"]["parameters"])
    volume = find_indicator_config(ind["VOLUME"]["name"], ind["VOLUME"]["parameters"])
    exit_indicator = find_indicator_config(
        ind["EXIT"]["name"], ind["EXIT"]["parameters"]
    )

    # Create the NNFXStrategy object
    strategy = strategies.NNFXStrategy(
        atr=atr,
        baseline=baseline,
        c1=c1,
        c2=c2,
        volume=volume,
        exit_indicator=exit_indicator,
        forex_pair=strategy_json["pair"],
        timeframe=strategy_json["timeframe"],
    )

    return strategy
