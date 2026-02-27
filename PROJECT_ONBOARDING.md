# Algo Trading Project Documentation (Onboarding Reference)

Last reviewed: 2026-02-25 (code inspection of current repository state)

This file is the reference-style project documentation.

Use the companion docs for task-specific guidance:

- `PROJECT_QUICKSTART.md` - step-by-step instructions to run the project properly
- `PROJECT_TODO.md` - improvement backlog and continuation ideas

## 1. Project Scope

This repository is a forex strategy research and execution codebase with:

- a multi-phase Optuna optimization/backtesting pipeline (`main.py`)
- a FastAPI UI wrapper for launching and monitoring runs (`backtest_ui.py`)
- a backtesting engine and SQLite result store (`scripts/backtester.py`, `scripts/data/sql.py`)
- strategy families:
  - `NNFX`
  - `Candlestick`
  - `Mabrouk2021` (ML-based)
- an optional OANDA demo/live trading loop (`trader.py`)

Primary output artifacts are:

- `backtesting/Backtesting.db` (central run/results DB)
- `backtesting/optuna_studies/*.db` (per-study Optuna DBs)
- caches under `backtesting/` and `machine_learning/`
- `best_strategies.json` (exported top-per-pair results)

## 2. Architecture Overview

## 2.1 Main execution surfaces

- `main.py`
  - batch optimizer/backtester
  - phase orchestration (phase1, phase2, phase3)
  - configured by editing constants in the `if __name__ == "__main__":` block
- `backtest_ui.py`
  - FastAPI UI + job manager
  - runs an internal worker subprocess that calls the same `main.py` pipeline functions
- `trader.py`
  - OANDA execution loop that reads `best_strategies.json`

## 2.2 Backtesting/optimization flow

Typical backtest/optimization flow:

1. Load OHLCV from per-pair SQLite DBs in `data/`.
2. Build strategy instances through adapter + strategy registries.
3. Run backtests and compute metrics.
4. Compute composite score.
5. Persist metrics, scores, and study metadata to `backtesting/Backtesting.db`.
6. Optionally export `best_strategies.json` from DB results.

## 2.3 Registry pattern (core extension mechanism)

The project uses two registries:

- Strategy registry (`scripts/strategies/strategy_core.py`)
  - `@register_strategy(...)`
  - `create_strategy_from_kwargs(...)`
  - `BaseStrategy`
- Trial adapter registry (`scripts/trial_adapters/base_adapter.py`)
  - `ADAPTER_REGISTRY`
  - `StrategyTrialAdapter`
  - shared trial lifecycle in `run_objective_common(...)`

This is the key design pattern for adding new strategies to the optimization pipeline.

## 3. Repository Layout (Practical Map)

Start here first:

- `main.py` - batch run orchestration
- `backtest_ui.py` - easiest controlled way to launch runs
- `scripts/config.py` - paths, pairs, timeframes, date windows, constants
- `scripts/backtester.py` - execution loop, metrics, composite score
- `scripts/data/sql.py` - DB schema and queries
- `scripts/strategies/` - strategy classes
- `scripts/trial_adapters/` - Optuna search spaces and phase builders

Important runtime directories:

- `data/`
  - one SQLite DB per pair (for example `EURUSD.db`)
- `backtesting/`
  - `Backtesting.db`
  - `optuna_studies/`
  - `indicator_caches/`
  - `ui_jobs/`
- `machine_learning/`
  - feature/label/model/score caches

Supporting/advanced areas:

- `scripts/indicators/` - indicator calculations, signal functions, indicator catalog
- `scripts/generated_indicators/` - generated indicator modules
- `manual_scripts/` - maintenance/codegen utilities
- `machine_learning/prev/` and `scripts/backup/` - legacy/backup code

## 4. Runtime Environment and Dependencies

## 4.1 Platform assumptions

The current codebase is Windows-oriented:

- `winsound` is imported in `main.py` and `trader.py`
- `backtest_ui.py` includes Windows-specific process stop handling (`taskkill`, process group flags)

Windows is the path of least resistance for current usage.

## 4.2 Python version

Recommended:

- Python 3.11 (3.10+ should work)

Reason:

- modern type syntax is used throughout (for example `dict[str, Any]`, `dict | None`)

## 4.3 Dependencies (project state)

A dependency manifest was not found in the repo root during review (no `requirements.txt` / `pyproject.toml` detected).

The codebase imports, at minimum, packages in these categories:

- numeric/data: `pandas`, `numpy`
- optimization: `optuna`
- UI/API: `fastapi`, `uvicorn`
- charting: `matplotlib`
- market data: `polygon-api-client` (`polygon`)
- progress/network helpers: `tqdm`, `urllib3`
- indicators: `TA-Lib` (`talib`)
- live trading: `oandapyV20`, `pytz`, `pyttsx3`
- ML/caching: `scikit-learn`, `joblib`, `pyarrow`

See `PROJECT_QUICKSTART.md` for a practical bootstrap command sequence.

## 5. Configuration Reference (`scripts/config.py`)

`scripts/config.py` is the central configuration module for paths, market universe, date windows, and key constants.

## 5.1 Paths and auto-created directories

Configured paths include:

- `DATA_FOLDER`
- `BACKTESTING_FOLDER`
- `OPTUNA_STUDIES_FOLDER`
- `TEMP_CACHE_FOLDER`
- ML cache directories:
  - `FEATURES_CACHE_DIR`
  - `LABELS_CACHE_DIR`
  - `SCORES_CACHE_DIR`
  - `MODELS_CACHE_DIR`

The module creates these directories on import.

## 5.2 Market universe and timeframes

Key lists:

- `MAJOR_FOREX_PAIRS`
- `ALL_TIMEFRAMES`
- `NNFX_TIMEFRAMES`

These values feed both direct runs (`main.py`) and UI defaults (`backtest_ui.py`).

## 5.3 Date windows used by optimization and evaluation

The project uses per-timeframe date windows in:

- `TIMEFRAME_DATE_RANGES_PHASE_1_AND_2`
- `TIMEFRAME_DATE_RANGES_PHASE3`

These are derived from:

- `START_DATE`
- `END_DATE`

Important current state:

- `END_DATE` is hardcoded to `2025-05-07` in `scripts/config.py`

This means runs can ignore newer data unless the config is updated.

## 5.4 Phase and scoring-related constants

Examples:

- `N_STARTUP_TRIALS_PERCENTAGE`
- `PHASE2_TOP_PERCENT`
- `MIN_TRADES_PER_DAY`
- pruning/scoring constants used by `scripts/backtester.py` and adapter code

## 6. Data Storage Model

## 6.1 Historical market data (`data/*.db`)

Historical data is stored as one SQLite DB per forex pair.

Examples:

- `data/EURUSD.db`
- `data/USDJPY.db`

Expected timeframe tables include names like:

- `5_minute`
- `15_minute`
- `30_minute`
- `1_hour`
- `2_hour`
- `4_hour`
- `1_day`

Reader/writer helper:

- `HistoricalDataSQLHelper` in `scripts/data/sql.py`

## 6.2 Central backtesting results DB (`backtesting/Backtesting.db`)

Managed by `BacktestSQLHelper` in `scripts/data/sql.py`.

Core tables:

- `ForexPairs`
- `StrategyConfigurations`
- `BacktestRuns`
- `Trades`
- `CompositeScores`
- `StudyMetadata`

Purpose by table:

- `StrategyConfigurations`
  - canonical serialized strategy configs for dedupe/reuse
- `BacktestRuns`
  - one-row metrics snapshots per run window
- `Trades`
  - trade-level rows linked to a backtest run
- `CompositeScores`
  - optimization objective values linked to runs/studies
- `StudyMetadata`
  - Optuna study summaries

## 6.3 Optuna study databases

Each study is stored as its own SQLite DB under:

- `backtesting/optuna_studies/`

Study names encode:

- phase
- pair
- timeframe
- strategy key
- evaluation window
- exploration space
- seed

## 6.4 Cache artifacts

Indicator caches:

- `backtesting/indicator_caches/`
- indexed by `IndicatorCacheSQLHelper`

ML caches (configured in `scripts/config.py`):

- `machine_learning/features_cache`
- `machine_learning/labels_cache`
- `machine_learning/models_cache`
- `machine_learning/scores_cache`

## 6.5 UI job artifacts

UI-launched runs create per-job directories under:

- `backtesting/ui_jobs/<job_id>/`

Typical files:

- `config.json`
- `run.log`

## 7. Backtesting and Optimization Pipeline (Reference)

## 7.1 Phase model

The project uses a 3-phase workflow:

- Phase 1
  - broad/default exploration
- Phase 2
  - parameterized exploration seeded from top phase-1 results
- Phase 3
  - fixed evaluation of top discovered strategy configurations

Phase-2 and phase-3 selection logic is implemented in `BacktestSQLHelper` query helpers.

## 7.2 Strategy adapter responsibilities

Each adapter defines:

- how to sample/build a strategy in Optuna (`objective(...)`)
- how phase 1/2/3 job arguments are generated (`build_phase1_args`, `build_phase2_args`, `build_phase3_args`)

Current adapters:

- `scripts/trial_adapters/nnxf_adapter.py`
- `scripts/trial_adapters/candlestick_adapter.py`
- `scripts/trial_adapters/mabrouk2021_adapter.py`

Current support note:

- `Candlestick` phase 3 adapter method currently returns no jobs (phase-3 support is not implemented there)

## 7.3 Shared trial lifecycle (dedupe, prune, persist)

The shared adapter path (`run_objective_common(...)`) handles:

- strategy config dedupe
- loading data and preparing indicators/features
- running `Backtester`
- validation and pruning (duplicate configs, low activity, invalid score, guardrails)
- queueing results for the DB writer process

This is the reason many Optuna trials may be pruned even when the system is working normally.

## 7.4 DB writer process pattern

`main.py` uses a multiprocessing DB writer process that consumes queue messages for:

- strategy config inserts
- indicator cache metadata inserts
- trial/run score inserts
- study metadata inserts
- phase-3 fixed run inserts

This design helps isolate SQLite writes from worker processes and reduces lock contention.

## 8. Results Model and Interpretation (Reference)

## 8.1 Result layers

There are three useful levels of results:

- Study-level (`StudyMetadata`)
  - Optuna run summary (best score/trial, counts, timing, score statistics)
- Backtest-run metrics (`BacktestRuns`)
  - detailed performance/risk/activity metrics for a run window
- Composite score (`CompositeScores`)
  - optimization ranking score used for selection/pruning

## 8.2 Key backtest metrics to understand

Important fields in `BacktestRuns` include:

- `Total_Trades`
- `Win_Rate` (stored as a fraction, not a 0-100 percent integer)
- `Net_Profit`
- `Total_Return_Pct`
- `Profit_Factor`
- `Max_Drawdown_Pct`
- `Trades_Per_Day`
- `Trade_Expectancy_Pct`
- `Expectancy_Per_Day_Pct`
- `Trade_Return_Std`

Metric-window detail:

- backtests may load a larger data window than the metrics/scoring window
- metrics can be computed on a sub-window (important for ML and evaluation-window logic)

## 8.3 Composite score (what it is)

Composite score is calculated in `scripts/backtester.py`.

It is not pure PnL. It combines:

- expectancy/day
- expectancy/trade
- profit factor
- win/loss asymmetry
- activity
- net profit contribution

And it applies penalties for:

- low activity relative to timeframe expectations
- high drawdown
- high return volatility

Use it as a ranking heuristic, not a standalone deployment decision.

## 9. Extension Points for Contributors (Reference)

## 9.1 Adding a strategy

Main files to study first:

- `scripts/strategies/strategy_core.py`
- `scripts/strategies/nnfx_strategy.py`
- `scripts/strategies/candlestick_strategy.py`
- `scripts/strategies/machine_learning_base.py`
- `scripts/trial_adapters/base_adapter.py`

In practice, adding a new optimizable strategy usually requires:

- a strategy class (registered with `@register_strategy`)
- a trial adapter (registered with `register_adapter(...)`)

Useful internal checklist:

- `scripts/strategies/new_strategy_checklist.txt`

## 9.2 Adding indicators

Indicator system entry points:

- `scripts/indicators/indicator_configs.py`
- `scripts/indicators/calculation_functions.py`
- `scripts/indicators/signal_functions.py`

The indicator catalog is used by:

- strategy code (NNFX/Candlestick)
- adapter search spaces
- ML feature pool construction (`machine_learning/core.py`, Mabrouk adapter)

## 9.3 ML strategy path

Main ML pipeline files:

- `machine_learning/core.py`
- `scripts/strategies/machine_learning_base.py`
- `scripts/strategies/mabrouk_2021.py`
- `scripts/trial_adapters/mabrouk2021_adapter.py`

Key ML design concepts in this codebase:

- leakage-safe feature shifting
- cached features/labels/models/scores
- time-aware splitting
- score-to-position-to-TradePlan mapping

## 10. Live Trading Module (Reference and Current Caveats)

`trader.py` is an OANDA polling/execution loop that:

- reads `best_strategies.json`
- reconstructs strategies
- polls candles
- generates `TradePlan` actions
- logs actions via `OandaSQLHelper`

Current state notes (factual):

- OANDA credentials are currently defined in `scripts/config.py`
- `trader.py` imports `load_strategy_from_dict` from `scripts.utilities`, but the function is defined in `scripts/strategies/strategy_core.py`
- `load_strategy_from_dict(...)` in `strategy_core.py` currently calls `create_strategy_from_kwargs(...)` with positional arguments, but the factory is keyword-based
- `trader.py` timeframe-to-granularity mapping only covers a subset of timeframes (`1_day`, `4_hour`, `2_hour`)

Treat live trading as a separate hardening effort, not a guaranteed turnkey path.

## 11. Current Known Limitations and Caveats (Documentation)

This section documents current codebase state and known caveats. It is intentionally descriptive, not a work plan.

- No dependency manifest found in repo root during review
- Windows-specific assumptions (`winsound`, process stop handling)
- `scripts/config.py` date windows are anchored to `END_DATE = 2025-05-07`
- `scripts/data/polygon.py` contains hardcoded API key usage in `collect_pair_data(...)`
- `scripts/data/polygon.py` `filter_collected_data_dates(...)` has a suspicious tail-range branch (the computed tail-start variable is not used in the returned range)
- `Candlestick` adapter phase-3 builder returns no jobs
- `trader.py` / `load_strategy_from_dict(...)` path and factory-call mismatch likely break live trading strategy reconstruction in current form

## 12. Related Docs in This Repo

- `PROJECT_QUICKSTART.md` - runbook/how-to with concrete knobs and step-by-step instructions
- `PROJECT_TODO.md` - improvement backlog and continuation roadmap
