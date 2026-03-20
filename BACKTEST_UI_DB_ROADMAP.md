# Backtest UI Database Roadmap

This file captures the implementation plan for database interaction features in `backtest_ui.py`.

## Phase 1 (Immediate, High-Value)

1. Database Health panel
   - DB file path, exists, size, modified time
   - table row counts
   - basic SQLite pragmas

2. Table Explorer
   - list of tables
   - schema/columns for selected table
   - row count and paged sample rows

3. Backtest Runs browser
   - filters: pair, timeframe, strategy, date range, score range, phase
   - server-side sort and pagination
   - drill-down actions (trades, strategy detail, resume config)

4. Study Metadata browser
   - filters: exploration space, stop reason, trials, score
   - sort and pagination

5. Composite Scores leaderboard
   - filters + sortable leaderboard
   - open linked backtest run

6. Trade drill-down
   - paged trades for selected backtest run
   - run summary context

7. Top Strategies tools
   - top percent by exploration space
   - top N across studies
   - best per pair/timeframe view

8. Strategy detail panel
   - parsed strategy parameters
   - linked runs and best scores

9. Export tools
   - CSV/JSON export for current visible datasets

10. Saved Views
   - persist filter presets per DB tab (browser local storage)

11. One-click resume from DB selection
   - generate start payload from a selected backtest run
   - direct "run now" action

12. Job-to-DB traceability
   - capture study names touched by each UI job
   - show studies/backtests linked to a selected job id

## Phase 2 (Operations + Data Quality)

1. Integrity checks
   - orphan rows
   - broken foreign key relationships
   - missing/invalid metric anomalies

2. Maintenance workflows
   - dry-run cleanup actions
   - explicit confirmation gates

3. Coverage analytics
   - pair/timeframe heatmaps
   - stale or under-tested combo detection

4. Recompute utilities
   - score backfills and derived metric refresh

## Phase 3 (Advanced Analytics + Governance)

1. Side-by-side run comparison
2. Parameter diffing across strategy configs
3. Pareto front views (return/drawdown/stability)
4. Equity reconstruction and decomposition
5. Candidate promotion workflow toward trading DB
6. Admin/audit controls for destructive actions

## API Surface (Target)

- `GET /api/db/health`
- `GET /api/db/schema`
- `GET /api/db/table`
- `GET /api/db/backtest_runs`
- `GET /api/db/studies`
- `GET /api/db/composite_scores`
- `GET /api/db/trades`
- `GET /api/db/strategy/{strategy_id}`
- `GET /api/db/top_strategies`
- `GET /api/db/run_config/{backtest_id}`
- `GET /api/db/trace/{job_id}`

## Performance Rules

1. All large datasets must be server-paged.
2. Sorting must be limited to allowlisted columns.
3. Exports should use current filtered datasets.
4. Read-only DB access by default for browsing endpoints.
5. UI should avoid large payload redraws on every poll.
