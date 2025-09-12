# file: tools/smoke_mabrouk_ready.py
import importlib
import types
from typing import Iterable

import numpy as np
import pandas as pd

# --- Adjust these to your project layout ---
REGISTRY_PATH = "main"  # where your STRATEGY_REGISTRY lives
STRATEGY_KEY = "mabrouk_2021"  # your desired key in the registry

REQUIRED_OHLCV = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _make_dummy_data(n=300):
    ts = pd.date_range("2020-01-01", periods=n, freq="H")
    df = pd.DataFrame(
        {
            "symbol": "GBPUSD",
            "timestamp": ts,
            "open": np.linspace(1.30, 1.31, n) + np.random.randn(n) * 1e-3,
            "high": lambda x: x["open"] + np.abs(np.random.randn(n)) * 1e-3,
            "low": lambda x: x["open"] - np.abs(np.random.randn(n)) * 1e-3,
            "close": np.linspace(1.30, 1.305, n) + np.random.randn(n) * 1e-3,
            "volume": np.random.randint(100, 500, n),
        }
    )
    return df[REQUIRED_OHLCV]


def assert_columns(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"


def main():
    reg_mod = importlib.import_module(REGISTRY_PATH)
    registry = getattr(reg_mod, "STRATEGY_REGISTRY", None)
    assert isinstance(registry, dict), "No STRATEGY_REGISTRY dict found."

    assert (
        STRATEGY_KEY in registry
    ), f"{STRATEGY_KEY} not registered in STRATEGY_REGISTRY"
    strat_factory = registry[STRATEGY_KEY]
    strategy = strat_factory() if callable(strat_factory) else strat_factory

    # 1) Prepare data
    raw = _make_dummy_data()
    assert_columns(raw, REQUIRED_OHLCV)

    prepared = strategy.prepare(raw.copy())
    assert isinstance(prepared, pd.DataFrame), "prepare() must return a DataFrame"

    # label column optional; if present, check type
    if "label" in prepared.columns:
        assert pd.api.types.is_integer_dtype(
            prepared["label"]
        ) or pd.api.types.is_bool_dtype(
            prepared["label"]
        ), "label column must be integer/binary"

    # 2) Fit model (if strategy has fit)
    if hasattr(strategy, "fit"):
        split = int(len(prepared) * 0.7)
        train_df = prepared.iloc[:split]
        test_df = prepared.iloc[split:]
        strategy.fit(train_df)

        # 3) Predict
        preds = strategy.predict(test_df)
        assert hasattr(preds, "__len__") and len(preds) == len(
            test_df
        ), "predict must align with test_df length"
        # 4) Convert to trade plans
        if hasattr(strategy, "to_trade_plans"):
            plans = strategy.to_trade_plans(test_df, preds)
            assert isinstance(plans, list), "to_trade_plans must return a list"
            # Quick structural check (loose to allow your fields)
            for p in plans[:5]:
                assert hasattr(p, "entry_price") or isinstance(
                    p, dict
                ), "TradePlan missing entry_price (or use dict)"
        elif hasattr(strategy, "generate_trade_plan"):
            # Row-wise fallback
            plans = []
            for _, row in test_df.assign(_pred=preds).iterrows():
                plan = strategy.generate_trade_plan(row)
                if plan is not None:
                    plans.append(plan)
            assert isinstance(
                plans, list
            ), "generate_trade_plan must yield TradePlans in a list context"
        else:
            raise AssertionError(
                "Strategy exposes neither to_trade_plans nor generate_trade_plan"
            )

    print("✅ Mabrouk strategy wiring looks compatible with backtesting path.")


if __name__ == "__main__":
    main()
