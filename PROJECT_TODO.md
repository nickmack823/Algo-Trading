# Algo Trading Project To-Do and Continuation Roadmap

Last reviewed: 2026-02-25

This file is the improvement backlog and continuation plan.

Companion docs:

- `PROJECT_QUICKSTART.md` - how to run the project now
- `PROJECT_ONBOARDING.md` - current architecture and reference documentation

## 1. Priorities at a Glance

Recommended order:

1. Security and secrets cleanup
2. Correctness fixes that block live trading and data maintenance
3. Reproducible environment setup
4. Test coverage and CI
5. Data pipeline and UX improvements
6. Research/reporting and long-term platform improvements

## 2. Critical Fixes (Confirmed or Highly Likely Issues)

## 2.1 Secrets in source code (critical)

Current state:

- OANDA API key/account ID are hardcoded in `scripts/config.py`
- Polygon API key is hardcoded in `scripts/data/polygon.py`

Why this matters:

- security exposure
- unsafe repo sharing
- difficult environment separation (dev/demo/prod)

To do:

- move secrets to environment variables
- add a `.env.example` (or config template) with placeholder names only
- update docs to describe required env vars
- rotate exposed credentials/keys

## 2.2 Fix live trading strategy reconstruction path (high)

Current state:

- `trader.py` imports `load_strategy_from_dict` from `scripts.utilities`
- function is defined in `scripts/strategies/strategy_core.py`
- `load_strategy_from_dict(...)` currently calls `create_strategy_from_kwargs(...)` with positional args, but the factory is keyword-based

Why this matters:

- `trader.py` is likely broken for real strategy reconstruction

To do:

- fix the import path in `trader.py`
- fix `load_strategy_from_dict(...)` to call `create_strategy_from_kwargs(key=..., **kwargs)`
- define and document the expected `best_strategies.json` reconstruction schema
- add a reconstruction smoke test

## 2.3 Patch incremental data backfill logic in `scripts/data/polygon.py` (high)

Current state:

- `filter_collected_data_dates(...)` has a suspicious "missing tail" branch where the computed tail start is not used in the returned range

Why this matters:

- incremental updates may skip or miscompute missing ranges

To do:

- patch the date-range return logic
- add unit tests for:
  - no existing data
  - full coverage
  - missing head only
  - missing tail only
  - malformed/gapped data behavior

## 2.4 Remove hardcoded `END_DATE` bottleneck or make it configurable (high)

Current state:

- `scripts/config.py` anchors date windows to a hardcoded `END_DATE = 2025-05-07`

Why this matters:

- new data can exist in `data/*.db` but be ignored by optimization/backtests

To do:

- parameterize date windows (env/config/UI)
- optionally derive `END_DATE` from data coverage
- expose evaluation windows in UI in a controlled way

## 3. Reproducibility and Setup (High Value)

## 3.1 Add a dependency manifest

To do:

- create `requirements.txt` or `pyproject.toml`
- pin versions for critical packages (Optuna, pandas, numpy, scikit-learn, TA-Lib, FastAPI, pyarrow)
- document TA-Lib installation separately by OS/Python version

## 3.2 Add environment bootstrap scripts

To do:

- add a Windows bootstrap script (venv + pip install)
- optionally add Linux/macOS script with conditional notes
- include "verify imports" smoke command

## 3.3 Configuration hygiene

To do:

- separate secrets from runtime tuning constants
- consider `config.local.py` or env var overrides
- document precedence rules if multiple config sources are added

## 4. Testing and CI (Major Gap)

## 4.1 Add smoke tests for core pipeline paths

Focus first on:

- strategy registry and factory
- adapter registry loading
- `Backtester` run on a tiny synthetic dataset
- DB schema creation and inserts (`BacktestSQLHelper`)
- UI worker start/stop lifecycle (`backtest_ui.py`)

## 4.2 Add correctness tests for metrics and scoring

Test cases:

- no trades
- all wins / all losses
- drawdown calculations on controlled trade sequences
- composite score monotonicity sanity checks (for obvious cases)

## 4.3 Add ML leak-safety tests

Focus on:

- feature shifting behavior
- label alignment
- train/test window boundaries
- no pre-test trade guard behavior in adapter flow

## 4.4 Add CI

To do:

- run lint + smoke tests on push/PR
- verify imports on supported Python versions
- fail fast on missing secrets for live-trading-only code paths (without exposing credentials)

## 5. Data Pipeline Improvements

## 5.1 Add a proper CLI for historical data collection

Current state:

- data collection requires `python -c "...collect_all_data()..."`

To do:

- add `argparse` CLI to `scripts/data/polygon.py` (or a new `data_cli.py`)
- support:
  - all pairs
  - selected pairs
  - selected timeframes
  - from/to dates
  - dry-run mode

## 5.2 Add data validation and coverage reporting

To do:

- report latest timestamp per pair/timeframe
- detect missing tables
- detect sparse or stale DBs
- export a coverage summary to CSV/JSON

## 5.3 Improve data source and secret handling

To do:

- move Polygon key to env var
- support alternative providers or provider abstraction
- add rate-limit/backoff configuration

## 6. Backtesting and Optimization Reliability

## 6.1 SQLite write robustness and observability

To do:

- improve visibility into lock contention and retries
- add queue depth and throughput logging in the DB writer process
- add structured run summaries at end of each phase

## 6.2 Checkpointing and resumability UX

Current state:

- Optuna studies resume via `load_if_exists=True`
- study metadata skip behavior exists

To do:

- document and expose resume controls in UI
- add "resume / rerun / force rerun" options
- add explicit phase-level summary pages/logs

## 6.3 Phase-3 support consistency

To do:

- implement `Candlestick` phase-3 adapter builder (currently returns no jobs)
- verify phase-3 behavior parity across all supported strategy families

## 7. UI and Usability Improvements (`backtest_ui.py`)

## 7.1 Better presets and templates

To do:

- add saved run presets (quick sanity, moderate run, full run)
- add one-click "single-pair smoke test"
- add validation hints in the form UI for phases and dependencies

## 7.2 Richer progress and results visibility

To do:

- show per-study progress counts
- show estimated remaining work
- show summary stats after completion
- link to job artifacts and study DB names

## 7.3 Safer job management

To do:

- add queued jobs (currently only one active job is allowed)
- add confirmation for stop/terminate
- add explicit rerun action using a previous `config.json`

## 8. Research and Analytics Improvements

## 8.1 Result reporting and exports

To do:

- add standardized CSV/JSON exports for:
  - top runs
  - study summaries
  - phase comparison summaries
- add a small results dashboard/notebook template

## 8.2 Out-of-sample and walk-forward reporting

To do:

- formalize train/test/generalization reporting across adapters
- persist evaluation window metadata more explicitly in outputs
- add per-pair/per-timeframe validation report artifacts

## 8.3 Score transparency and tuning

To do:

- document composite score versioning
- store score components alongside final score
- experiment with alternative score formulas without losing comparability

## 9. Live Trading Hardening (Separate Workstream)

Treat this as its own project after backtesting reliability is solid.

To do:

- fix strategy reconstruction (Section 2.2)
- add dry-run / paper simulation mode independent of OANDA execution
- cap max concurrent strategies and per-pair exposure
- add circuit breakers / risk limits
- persist and recover position state across restarts
- add structured alerting and error reporting
- support broader timeframe granularity mapping if needed

## 10. Portability and Engineering Hygiene

## 10.1 Reduce Windows-only assumptions

To do:

- wrap `winsound` imports with platform guards
- use optional notifications interface instead of direct `winsound`
- verify `backtest_ui.py` stop behavior on non-Windows

## 10.2 Code organization and cleanup

To do:

- separate legacy/backup code from active code paths more clearly
- reduce large monolithic modules where practical (`main.py`, `machine_learning/core.py`)
- centralize duplicated constants and helper patterns

## 10.3 Documentation set expansion

Possible additions:

- `SETUP.md` (pinned environment setup)
- `CONTRIBUTING.md` (workflow, coding expectations, test requirements)
- `DB_SCHEMA.md` (tables and relationships)
- `LIVE_TRADING.md` (once hardened)

## 11. Suggested Next Milestone (Pragmatic)

If you want the highest near-term value with low risk, do this next:

1. Remove secrets from code and rotate keys.
2. Add `requirements.txt` (or `pyproject.toml`) and TA-Lib install notes.
3. Fix `trader.py` + `load_strategy_from_dict(...)`.
4. Patch and test `filter_collected_data_dates(...)`.
5. Add one smoke test for the UI -> worker -> DB path on a tiny dataset.

After that, the project becomes much easier to operate and extend safely.
