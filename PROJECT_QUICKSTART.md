# Algo Trading Quickstart (How to Use)

Last reviewed: 2026-02-25

This is the practical runbook for getting the project running correctly.

Companion docs:

- `PROJECT_ONBOARDING.md` - architecture and reference documentation
- `PROJECT_TODO.md` - improvement backlog and continuation ideas

## 1. What to use this for

Use this file when you want to:

- set up the environment
- verify data is present
- run a safe first backtest/optimization job
- scale up runs using the right knobs

The recommended path is the FastAPI UI in `backtest_ui.py`.

## 2. Before You Start (Required)

## 2.1 OS and Python

Recommended:

- Windows
- Python 3.11 (3.10+ should work)

## 2.2 Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2.3 Install dependencies (practical baseline)

There is no pinned dependency file in the repo right now, so install the core stack manually.

```powershell
pip install pandas numpy optuna fastapi uvicorn matplotlib tqdm urllib3 polygon-api-client oandapyV20 pyttsx3 pytz joblib scikit-learn pyarrow
```

Important extra dependency:

- Install TA-Lib (`talib`) before trying to run strategies.

If `import talib` fails, the project will fail early when loading indicator configs.

## 2.4 Verify historical data exists

Check that `data/` contains pair DBs (for example `EURUSD.db`) and that the DB has timeframe tables.

Quick checks:

```powershell
Get-ChildItem data | Select-Object -First 10
```

If `data/` is missing or mostly empty, see Section 7 (data collection).

## 3. Recommended First Run (FastAPI UI)

## 3.1 Start the UI server

```powershell
python backtest_ui.py serve --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000`

## 3.2 Run a small sanity-check job (safe settings)

Use these settings in the UI for a first run:

- Strategies: `Mabrouk2021` (or `NNFX`)
- Phase 1: enabled
- Phase 2: disabled
- Phase 3: disabled
- Pairs: `EUR/USD`
- Phase1/2 Timeframes: `2_hour`
- Phase3 Timeframes: blank
- Seeds: `42`
- Parallel Processes (`n_processes`): `1`
- Phase2 Top Percent: leave default
- Phase3 Top N: leave default
- Trials By Timeframe:
  - `2_hour=10`

Then click `Start`.

Why this is the right first run:

- single pair
- single timeframe
- single seed
- low trial count
- single process (easier debugging, fewer SQLite lock issues)

## 3.3 Monitor the run

In the UI:

- watch Active Job status
- watch log tail for progress and errors
- confirm the job ends as `completed`

Artifacts written by the UI job manager:

- `backtesting/ui_jobs/<job_id>/config.json`
- `backtesting/ui_jobs/<job_id>/run.log`

## 3.4 Verify outputs after the run

Check these locations:

- `backtesting/Backtesting.db`
- `backtesting/optuna_studies/` (should contain at least one study DB)
- `backtesting/ui_jobs/<job_id>/run.log`

If the run completes cleanly, you have validated:

- environment and imports
- data loading
- strategy/adapters
- backtester execution
- Optuna study creation
- DB writes

## 4. The Knobs That Matter (UI and Worker Config)

These map to the UI form and the worker payload in `backtest_ui.py`.

## 4.1 Strategy and phase knobs

- `strategies`
  - which adapters/strategy families run
  - valid values come from the adapter registry (UI loads them dynamically)
- `enable_phase1`
  - broad/default exploration
- `enable_phase2`
  - parameterized follow-up search (depends on phase-1 results existing)
- `enable_phase3`
  - fixed evaluation of top strategies (depends on prior results)

Practical rule:

- Start with phase 1 only
- Add phase 2 after phase-1 results exist
- Add phase 3 only when you actually want fixed evaluation/export

## 4.2 Pair and timeframe knobs

- `pairs`
  - comma-separated list, for example `EUR/USD,USD/JPY`
- `phase12_timeframes`
  - timeframes used by phases 1 and 2
- `phase3_timeframes`
  - optional override for phase 3
  - blank means "all configured timeframes" in `main.py` phase-3 logic

Practical rule:

- Keep phases 1/2 on one timeframe until the pipeline is stable.

## 4.3 Search size knobs

- `seeds`
  - Optuna sampler seeds; more seeds = more study databases and longer runtime
- `trials_by_timeframe`
  - number of trials per timeframe for phases 1 and 2
- `n_processes`
  - parallel study count / worker concurrency

Practical rule for debugging:

- `seeds=42`
- `n_processes=1`
- low trial counts (`10` to `30`)

Practical rule for scaling:

- increase trials first
- then increase pairs/timeframes
- increase `n_processes` last (SQLite and system resources become the limiting factor)

## 4.4 Selection knobs

- `phase2_top_percent`
  - percentage of phase-1 strategies retained for phase-2 exploration
- `phase3_top_n`
  - number of top unique strategies selected for fixed evaluation in phase 3

Practical rule:

- leave defaults until you trust the full pipeline behavior

## 5. Scaling Up Safely (After the Sanity Run)

Use this progression:

1. One pair, one timeframe, phase 1 only, 10-30 trials.
2. One pair, one timeframe, phase 1 + phase 2.
3. One pair, one timeframe, add phase 3.
4. Multiple pairs on same timeframe.
5. Multiple timeframes.
6. Increase `n_processes` after observing stable DB writes and no import/data issues.

If anything fails, go back one step and reduce:

- pairs
- timeframes
- trials
- processes

## 6. Direct Run Mode (`main.py`) - Power User Path

If you prefer running without the UI:

```powershell
python main.py
```

Before running, edit the constants in the `if __name__ == "__main__":` block of `main.py`:

- `STRATEGY_KEYS_TO_RUN`
- `N_PROCESSES`
- `CURRENT_SEEDS`
- `TIMEFRAMES_PHASES_1_2`
- `TRIALS_BY_TIMEFRAME`
- `PHASES`
- `pairs` (defaults to `MAJOR_FOREX_PAIRS`)

Important:

- `main.py` does not use CLI flags for these settings.
- The UI is safer for repeatable runs because it persists job configs and logs.

## 7. If You Need to Populate Historical Data

Built-in data collection exists in `scripts/data/polygon.py`.

Run:

```powershell
python -c "from scripts.data.polygon import collect_all_data; collect_all_data()"
```

Notes:

- This is separate from `main.py`; backtest runs do not auto-fetch data.
- The code currently uses a hardcoded Polygon API key path in `collect_pair_data(...)`.
- Review `PROJECT_TODO.md` before relying on incremental backfill behavior.

## 8. Quick Troubleshooting (Most Common Failures)

## 8.1 `talib` import error

Symptoms:

- startup/import failure when indicator configs load

Fix:

- install TA-Lib correctly for your Python version and platform

## 8.2 "No such table" or missing data for timeframe

Symptoms:

- backtest fails when loading pair/timeframe

Fix:

- verify `data/<PAIR>.db` exists
- verify the requested table exists (for example `2_hour`)
- run data collection if needed

## 8.3 Runs stop earlier than expected / data looks stale

Cause:

- `scripts/config.py` currently hardcodes `END_DATE = 2025-05-07`

Fix:

- update `scripts/config.py` date windows to match your data coverage

## 8.4 Phase 3 does nothing for `Candlestick`

Cause:

- `Candlestick` adapter phase-3 builder currently returns no jobs

Fix:

- use `NNFX` or `Mabrouk2021` for phase-3 testing, or implement candlestick phase-3 support

## 8.5 Live trading (`trader.py`) does not work out of the box

Cause:

- current strategy reconstruction path has known mismatches

Fix:

- treat live trading as a separate hardening task (see `PROJECT_TODO.md`)

## 9. What "Done" Looks Like for a Proper First Setup

You are in good shape if all of the following are true:

- UI starts at `http://127.0.0.1:8000`
- a small phase-1 job completes successfully
- `backtesting/optuna_studies/` has a study DB
- `backtesting/Backtesting.db` has results
- UI `run.log` shows study progress and completion

After that, use `PROJECT_ONBOARDING.md` to understand the architecture and result interpretation before scaling up.
