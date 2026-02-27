from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse


ROOT = Path(__file__).resolve().parent
JOBS_ROOT = ROOT / "backtesting" / "ui_jobs"
LOG_TAIL_LINES = 300


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    out: list[int] = []
    for x in split_csv(text):
        try:
            out.append(int(x))
        except ValueError as exc:
            raise ValueError(f"Invalid integer: {x}") from exc
    return out


def parse_trials_text(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Invalid trials line '{line}', use timeframe=value")
        key, value = [p.strip() for p in line.split("=", 1)]
        if not key:
            raise ValueError(f"Missing timeframe in line '{line}'")
        try:
            n = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid trial count '{value}' for {key}") from exc
        if n <= 0:
            raise ValueError(f"Trial count must be > 0 for {key}")
        out[key] = n
    return out


def trials_to_text(trials: dict[str, int]) -> str:
    return "\n".join(f"{k}={v}" for k, v in trials.items())


def num_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def load_defaults() -> dict[str, Any]:
    defaults = {
        "available_strategies": ["Mabrouk2021", "NNFX", "Candlestick"],
        "available_timeframes": [
            "5_minute",
            "15_minute",
            "30_minute",
            "1_hour",
            "2_hour",
            "4_hour",
            "1_day",
        ],
        "default": {
            "strategies": ["Mabrouk2021"],
            "pairs": ["EUR/USD", "USD/JPY", "GBP/USD"],
            "phase12_timeframes": ["2_hour", "4_hour", "1_day"],
            "phase3_timeframes": [],
            "seeds": [42, 1337, 314, 777, 444],
            "n_processes": 1,
            "phase2_top_percent": 10.0,
            "phase3_top_n": 30,
            "enable_phase1": True,
            "enable_phase2": True,
            "enable_phase3": True,
            "trials_by_timeframe": {
                "1_day": 500,
                "4_hour": 300,
                "2_hour": 300,
                "1_hour": 200,
                "30_minute": 200,
                "15_minute": 200,
                "5_minute": 200,
            },
        },
    }
    try:
        from scripts.config import (
            ALL_TIMEFRAMES,
            MAJOR_FOREX_PAIRS,
            NNFX_TIMEFRAMES,
            PHASE2_TOP_PERCENT,
        )
        import main  # noqa: F401
        from scripts.trial_adapters.base_adapter import ADAPTER_REGISTRY

        defaults["available_timeframes"] = list(ALL_TIMEFRAMES)
        defaults["default"]["pairs"] = list(MAJOR_FOREX_PAIRS)
        defaults["default"]["phase12_timeframes"] = list(NNFX_TIMEFRAMES)
        defaults["default"]["phase2_top_percent"] = float(PHASE2_TOP_PERCENT)
        if ADAPTER_REGISTRY:
            defaults["available_strategies"] = list(ADAPTER_REGISTRY.keys())
            defaults["default"]["strategies"] = [
                "Mabrouk2021"
                if "Mabrouk2021" in ADAPTER_REGISTRY
                else defaults["available_strategies"][0]
            ]
    except Exception as exc:
        defaults["warning"] = f"Using fallback defaults ({type(exc).__name__}: {exc})"

    defaults["default"]["trials_by_timeframe_text"] = trials_to_text(
        defaults["default"]["trials_by_timeframe"]
    )
    return defaults


UI_DEFAULTS = load_defaults()


def normalize_start_payload(raw: dict[str, Any]) -> dict[str, Any]:
    available_strats = set(UI_DEFAULTS.get("available_strategies", []))
    available_tfs = set(UI_DEFAULTS.get("available_timeframes", []))

    strategies = raw.get("strategies", [])
    if not isinstance(strategies, list):
        strategies = split_csv(str(strategies))
    strategies = [str(x).strip() for x in strategies if str(x).strip()]
    if not strategies:
        raise ValueError("Select at least one strategy.")
    bad = [s for s in strategies if available_strats and s not in available_strats]
    if bad:
        raise ValueError("Unknown strategy keys: " + ", ".join(bad))

    pairs = raw.get("pairs", "")
    if isinstance(pairs, list):
        pairs = [str(x).strip() for x in pairs if str(x).strip()]
    else:
        pairs = split_csv(pairs)
    if not pairs:
        raise ValueError("Provide at least one pair.")

    phase12_tfs = raw.get("phase12_timeframes", "")
    if isinstance(phase12_tfs, list):
        phase12_tfs = [str(x).strip() for x in phase12_tfs if str(x).strip()]
    else:
        phase12_tfs = split_csv(phase12_tfs)

    phase3_tfs = raw.get("phase3_timeframes", "")
    if isinstance(phase3_tfs, list):
        phase3_tfs = [str(x).strip() for x in phase3_tfs if str(x).strip()]
    else:
        phase3_tfs = split_csv(phase3_tfs)

    enable_phase1 = bool(raw.get("enable_phase1"))
    enable_phase2 = bool(raw.get("enable_phase2"))
    enable_phase3 = bool(raw.get("enable_phase3"))
    if not any([enable_phase1, enable_phase2, enable_phase3]):
        raise ValueError("Enable at least one phase.")
    if (enable_phase1 or enable_phase2) and not phase12_tfs:
        raise ValueError("Provide phases 1/2 timeframes.")

    bad_tfs = [tf for tf in phase12_tfs + phase3_tfs if available_tfs and tf not in available_tfs]
    if bad_tfs:
        raise ValueError("Invalid timeframe(s): " + ", ".join(sorted(set(bad_tfs))))

    seeds = raw.get("seeds", "")
    try:
        if isinstance(seeds, list):
            seeds = [int(x) for x in seeds]
        else:
            seeds = parse_int_csv(str(seeds))
    except Exception as exc:
        raise ValueError("Seeds must be a comma-separated list of integers.") from exc
    if (enable_phase1 or enable_phase2) and not seeds:
        raise ValueError("Provide at least one seed.")

    trials = raw.get("trials_by_timeframe")
    try:
        if isinstance(trials, dict):
            trials = {str(k): int(v) for k, v in trials.items()}
        else:
            trials = parse_trials_text(str(raw.get("trials_by_timeframe_text", "")))
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    if enable_phase1 or enable_phase2:
        missing = [tf for tf in phase12_tfs if tf not in trials]
        if missing:
            raise ValueError("Missing trial counts for: " + ", ".join(missing))

    try:
        n_processes = int(raw.get("n_processes", 1))
    except Exception as exc:
        raise ValueError("n_processes must be an integer.") from exc
    if n_processes <= 0:
        raise ValueError("n_processes must be > 0")
    try:
        phase2_top_percent = float(raw.get("phase2_top_percent", 10))
    except Exception as exc:
        raise ValueError("phase2_top_percent must be a number.") from exc
    if not (0 < phase2_top_percent <= 100):
        raise ValueError("phase2_top_percent must be > 0 and <= 100")
    try:
        phase3_top_n = int(raw.get("phase3_top_n", 30))
    except Exception as exc:
        raise ValueError("phase3_top_n must be an integer.") from exc
    if phase3_top_n <= 0:
        raise ValueError("phase3_top_n must be > 0")

    return {
        "strategies": strategies,
        "pairs": pairs,
        "phase12_timeframes": phase12_tfs,
        "phase3_timeframes": phase3_tfs,
        "seeds": seeds,
        "trials_by_timeframe": trials,
        "n_processes": n_processes,
        "phase2_top_percent": phase2_top_percent,
        "phase3_top_n": phase3_top_n,
        "enable_phase1": enable_phase1,
        "enable_phase2": enable_phase2,
        "enable_phase3": enable_phase3,
        "created_at": now_iso(),
    }


@dataclass
class Job:
    job_id: str
    config: dict[str, Any]
    config_path: str
    log_path: str
    status: str = "running"
    created_at: str = field(default_factory=now_iso)
    started_at: Optional[str] = field(default_factory=now_iso)
    ended_at: Optional[str] = None
    pid: Optional[int] = None
    return_code: Optional[int] = None
    error: Optional[str] = None
    stop_requested: bool = False
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=2000), repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "pid": self.pid,
            "return_code": self.return_code,
            "error": self.error,
            "stop_requested": self.stop_requested,
            "config": {
                "strategies": self.config.get("strategies", []),
                "pairs_count": len(self.config.get("pairs", [])),
                "phase12_timeframes": self.config.get("phase12_timeframes", []),
                "phase3_timeframes": self.config.get("phase3_timeframes", []),
                "enable_phase1": self.config.get("enable_phase1", False),
                "enable_phase2": self.config.get("enable_phase2", False),
                "enable_phase3": self.config.get("enable_phase3", False),
                "n_processes": self.config.get("n_processes"),
            },
        }

    def tail(self) -> list[str]:
        return list(self.logs)[-LOG_TAIL_LINES:]


class JobManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)

    def active(self) -> Optional[Job]:
        with self.lock:
            for job_id in reversed(self.order):
                job = self.jobs[job_id]
                if job.status in {"running", "stopping"}:
                    return job
            return None

    def list_summaries(self) -> list[dict[str, Any]]:
        with self.lock:
            return [self.jobs[jid].summary() for jid in reversed(self.order)]

    def get(self, job_id: str) -> Job:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]

    def start(self, config: dict[str, Any]) -> Job:
        with self.lock:
            active = self.active()
            if active:
                raise RuntimeError(f"Job {active.job_id} is still {active.status}")

            job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            job_dir = JOBS_ROOT / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            config_path = job_dir / "config.json"
            log_path = job_dir / "run.log"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            log_path.write_text("", encoding="utf-8")

            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(config_path)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )

            job = Job(
                job_id=job_id,
                config=config,
                config_path=str(config_path),
                log_path=str(log_path),
                pid=proc.pid,
                process=proc,
            )
            self.jobs[job_id] = job
            self.order.append(job_id)
            threading.Thread(target=self._read_logs, args=(job_id,), daemon=True).start()
            threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()
            return job

    def _append_log(self, job_id: str, line: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job.logs.append(line.rstrip("\n"))
            log_path = Path(job.log_path)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def _read_logs(self, job_id: str) -> None:
        try:
            job = self.get(job_id)
            proc = job.process
            if proc is None or proc.stdout is None:
                return
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                self._append_log(job_id, line)
        except Exception as exc:
            self._append_log(job_id, f"[UI] log reader error: {exc}\n")

    def _watch(self, job_id: str) -> None:
        try:
            job = self.get(job_id)
            proc = job.process
            if proc is None:
                return
            rc = proc.wait()
            with self.lock:
                j = self.jobs.get(job_id)
                if not j:
                    return
                j.return_code = rc
                j.ended_at = now_iso()
                if j.stop_requested:
                    j.status = "stopped"
                else:
                    j.status = "completed" if rc == 0 else "failed"
        except Exception as exc:
            with self.lock:
                j = self.jobs.get(job_id)
                if j:
                    j.status = "failed"
                    j.error = f"watcher error: {exc}"
                    j.ended_at = now_iso()

    def stop(self, job_id: str) -> Job:
        with self.lock:
            job = self.get(job_id)
            proc = job.process
            if proc is None or proc.poll() is not None:
                return job
            job.stop_requested = True
            job.status = "stopping"
            pid = proc.pid

        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if os.name == "nt" and pid:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                proc.kill()
        return self.get(job_id)


JOBS = JobManager()
app = FastAPI(title="Backtesting UI")


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    defaults_json = json.dumps(UI_DEFAULTS).replace("</", "<\\/")
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtesting UI</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: Segoe UI, sans-serif; margin: 12px; overflow-x: hidden; }}
    .wrap {{ display:grid; grid-template-columns: minmax(0, 390px) minmax(0, 1fr); gap: 12px; align-items: start; }}
    .box {{ border:1px solid #ccc; border-radius:8px; padding:12px; }}
    label {{ display:block; margin-top:8px; font-size:12px; color:#444; }}
    input, textarea {{ width:100%; box-sizing:border-box; padding:6px; }}
    textarea {{ min-height:80px; }}
    .hint {{ margin-top:4px; font-size:11px; color:#666; line-height:1.4; }}
    .phase-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:8px; }}
    .btns {{ margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; }}
    pre {{ background:#111; color:#d7f7d7; padding:10px; min-height:220px; max-height:420px; overflow:auto; }}
    pre.log-pre {{
      white-space: pre-wrap;
      word-break: break-word;
      overflow-x: hidden;
      font-size:11px;
      line-height:1.3;
    }}
    pre.summary-pre {{
      background:#f7f9fc;
      color:#263242;
      border:1px solid #d7dee8;
      min-height:0;
      max-height:none;
      margin-top:6px;
      margin-bottom:0;
      font-size:11px;
      line-height:1.45;
    }}
    .metrics {{ margin-top:10px; }}
    .tabs {{ display:flex; gap:6px; margin-top:8px; margin-bottom:8px; }}
    .tab-btn {{
      border:1px solid #b8c3cf;
      background:#f4f8fc;
      color:#31465c;
      border-radius:999px;
      font-size:12px;
      padding:4px 10px;
      cursor:pointer;
    }}
    .tab-btn.active {{ background:#dfefff; border-color:#8bb3dd; font-weight:600; }}
    .tab-panel {{ display:none; }}
    .tab-panel.active {{ display:block; }}
    .guide-box {{
      border:1px solid #d7dee8;
      border-radius:8px;
      padding:8px 10px;
      background:#fbfdff;
    }}
    .guide-item {{ margin-top:6px; font-size:12px; color:#2f3d4e; line-height:1.4; }}
    table {{ width:100%; border-collapse: collapse; font-size:12px; table-layout: fixed; }}
    th, td {{ border-bottom:1px solid #ddd; padding:4px; text-align:left; word-break: break-word; }}
    @media (max-width: 1200px) {{ .phase-grid {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 1100px) {{ .wrap {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h2>main.py Backtesting UI</h2>
  <div id="warn" style="color:#8a5a00"></div>
  <div class="wrap">
    <div class="box">
      <div><b>Backtest Controls</b></div>
      <div class="hint">Use tabs to switch between setup and guidance without long scrolling.</div>
      <div class="tabs">
        <button id="tab_setup_btn" class="tab-btn active" onclick="switchTab('setup')">Setup</button>
        <button id="tab_guide_btn" class="tab-btn" onclick="switchTab('guide')">Guide</button>
      </div>

      <div id="tab_setup" class="tab-panel active">
        <label>Strategies</label>
        <div id="strategies"></div>
        <div class="hint">Choose one or more strategy adapters to include in the run.</div>
        <label>Phases</label>
        <div class="phase-grid">
          <label><input type="checkbox" id="phase1"> Phase 1</label>
          <label><input type="checkbox" id="phase2"> Phase 2</label>
          <label><input type="checkbox" id="phase3"> Phase 3</label>
        </div>
        <div class="hint">Run at least one phase. New users usually start with Phase 1 and Phase 2.</div>
        <label>Pairs (comma-separated)</label>
        <textarea id="pairs"></textarea>
        <div class="hint">Forex symbols such as EUR/USD, GBP/USD, USD/JPY. More pairs increase runtime.</div>
        <label>Phase1/2 Timeframes (comma-separated)</label>
        <input id="phase12_tfs">
        <div class="hint">Timeframes used for optimization phases. Example: 2_hour, 4_hour, 1_day.</div>
        <label>Phase3 Timeframes (optional, blank = all)</label>
        <input id="phase3_tfs">
        <div class="hint">Optional final validation filter. Leave blank to let Phase 3 consider all available timeframes.</div>
        <label>Seeds (comma-separated)</label>
        <input id="seeds">
        <div class="hint">Different random seeds test stability. More seeds improve confidence but take longer.</div>
        <label>Parallel Processes</label>
        <input id="n_processes" type="number" min="1" step="1">
        <div class="hint">How many studies can run at the same time. Start with 1 if unsure.</div>
        <label>Phase2 Top Percent</label>
        <input id="phase2_top_percent" type="number" min="0.1" max="100" step="0.1">
        <div class="hint">After Phase 1, keep only this top percent of candidates for Phase 2 tuning.</div>
        <label>Phase3 Top N</label>
        <input id="phase3_top_n" type="number" min="1" step="1">
        <div class="hint">Final shortlist size to evaluate deeply in Phase 3.</div>
        <label>Trials By Timeframe (one per line: timeframe=value)</label>
        <textarea id="trials_text" style="min-height:120px"></textarea>
        <div class="hint">Number of optimization attempts per timeframe for Phases 1/2. Higher values increase search depth and runtime.</div>
        <div class="metrics">
          <div><b>Run Size Estimate</b></div>
          <pre id="run_summary" class="summary-pre"></pre>
        </div>
        <div class="btns">
          <button onclick="startRun()">Start</button>
          <button onclick="stopRun()">Stop Active</button>
          <button onclick="resetForm()">Reset</button>
          <button onclick="refreshState()">Refresh</button>
        </div>
        <div class="hint">
          <b>Start:</b> starts a new run with current settings. <b>Stop Active:</b> requests stop for current run.
          <b>Reset:</b> restores default form values only (does not stop jobs). <b>Refresh:</b> reloads active logs and job tables now.
        </div>
      </div>

      <div id="tab_guide" class="tab-panel">
        <div class="guide-box">
          <div class="guide-item"><b>Phase 1:</b> broad search over each strategy, pair, timeframe, and seed.</div>
          <div class="guide-item"><b>Phase 2:</b> narrows to top performers from Phase 1 and tunes around them.</div>
          <div class="guide-item"><b>Phase 3:</b> final ranking/validation pass for the best candidates.</div>
          <div class="guide-item"><b>Starter recipe:</b> keep defaults, use 2-3 timeframes, and run with <code>n_processes=1</code> first.</div>
          <div class="guide-item"><b>Common validation error:</b> every Phase1/2 timeframe must appear in the trials list.</div>
          <div class="guide-item"><b>Reset vs Refresh:</b> Reset changes form inputs back to defaults; Refresh just fetches latest status/logs.</div>
        </div>
      </div>
      <div id="msg" style="margin-top:8px;font-size:12px;"></div>
    </div>
    <div style="display:grid; gap:16px;">
      <div class="box">
        <div><b>Active Job</b> <span id="activeMeta"></span></div>
        <pre id="logs" class="log-pre">No active job.</pre>
      </div>
      <div class="box">
        <div><b>Resumable Runs</b> <span class="hint">(stopped or failed)</span></div>
        <table>
          <thead><tr><th>Job</th><th>Phases</th><th>Strategies</th><th>Status</th><th>Ended</th><th>Resume</th></tr></thead>
          <tbody id="resumable"></tbody>
        </table>
      </div>
      <div class="box">
        <div><b>History</b></div>
        <table><thead><tr><th>Job</th><th>Status</th><th>PID</th><th>Phases</th><th>Strategies</th></tr></thead><tbody id="history"></tbody></table>
      </div>
    </div>
  </div>
<script>
const DEF = {defaults_json};
let timer = null;

function setMsg(text, err=false) {{
  const el = document.getElementById('msg');
  el.textContent = text || '';
  el.style.color = err ? '#b00020' : '#1d5d28';
}}

function switchTab(name) {{
  const setup = name === 'setup';
  document.getElementById('tab_setup').classList.toggle('active', setup);
  document.getElementById('tab_guide').classList.toggle('active', !setup);
  document.getElementById('tab_setup_btn').classList.toggle('active', setup);
  document.getElementById('tab_guide_btn').classList.toggle('active', !setup);
}}

function csvItems(text) {{
  return String(text || '')
    .split(',')
    .map(x => x.trim())
    .filter(Boolean);
}}

function parseTrialsLines(text) {{
  const map = {{}};
  const invalid = [];
  for (const rawLine of String(text || '').split('\\n')) {{
    const line = rawLine.trim();
    if (!line) continue;
    const [tf, n] = line.split('=', 2).map(x => x.trim());
    const value = Number(n);
    if (!tf || !Number.isInteger(value) || value <= 0) {{
      invalid.push(line);
      continue;
    }}
    map[tf] = value;
  }}
  return {{ map, invalid }};
}}

function bindStrategyHandlers() {{
  for (const cb of document.querySelectorAll('.strategy')) {{
    cb.addEventListener('change', updateRunSummary);
  }}
}}

function renderStrategies() {{
  const wrap = document.getElementById('strategies');
  wrap.innerHTML = '';
  for (const s of (DEF.available_strategies || [])) {{
    const id = 's_' + s;
    wrap.innerHTML += `<label><input type="checkbox" class="strategy" value="${{s}}" id="${{id}}"> ${{s}}</label>`;
  }}
  bindStrategyHandlers();
}}

function updateRunSummary() {{
  const selectedStrategies = document.querySelectorAll('.strategy:checked').length;
  const pairs = csvItems(document.getElementById('pairs').value);
  const phase12 = csvItems(document.getElementById('phase12_tfs').value);
  const phase3 = csvItems(document.getElementById('phase3_tfs').value);
  const seeds = csvItems(document.getElementById('seeds').value);
  const trials = parseTrialsLines(document.getElementById('trials_text').value);
  const phase1 = document.getElementById('phase1').checked;
  const phase2 = document.getElementById('phase2').checked;
  const phase3Enabled = document.getElementById('phase3').checked;

  const lines = [];
  lines.push(`Selected strategies: ${{selectedStrategies}}`);
  lines.push(`Pairs: ${{pairs.length}} | Seeds: ${{seeds.length}} | Phase1/2 TFs: ${{phase12.length}}`);
  lines.push(`Phase3 TF override: ${{phase3.length ? phase3.length : 'blank (all available)'}}`);

  const enabledPhases = [];
  if (phase1) enabledPhases.push('Phase 1');
  if (phase2) enabledPhases.push('Phase 2');
  if (phase3Enabled) enabledPhases.push('Phase 3');
  lines.push(`Enabled phases: ${{enabledPhases.length ? enabledPhases.join(', ') : 'none'}}`);

  const missing = phase12.filter(tf => !(tf in trials.map));
  if (missing.length) {{
    lines.push(`Missing trial values for: ${{missing.join(', ')}}`);
  }} else if (phase12.length && seeds.length && pairs.length) {{
    const trialsPerSeedPair = phase12.reduce((sum, tf) => sum + (trials.map[tf] || 0), 0);
    const approxJobs = selectedStrategies * pairs.length * seeds.length * trialsPerSeedPair;
    lines.push(`Approx Phase1/2 trial jobs total: ${{approxJobs.toLocaleString()}}`);
  }} else {{
    lines.push('Approx Phase1/2 trial jobs total: unavailable (need strategies, pairs, seeds, and phase1/2 timeframes).');
  }}

  if (trials.invalid.length) {{
    lines.push(`Invalid trial lines: ${{trials.invalid.length}}`);
  }}
  lines.push('Tip: run a smaller setup first, then scale up once logs look healthy.');
  document.getElementById('run_summary').textContent = lines.join('\\n');
}}

function resetForm() {{
  renderStrategies();
  const d = DEF.default || {{}};
  for (const cb of document.querySelectorAll('.strategy')) {{
    cb.checked = (d.strategies || []).includes(cb.value);
  }}
  document.getElementById('phase1').checked = !!d.enable_phase1;
  document.getElementById('phase2').checked = !!d.enable_phase2;
  document.getElementById('phase3').checked = !!d.enable_phase3;
  document.getElementById('pairs').value = (d.pairs || []).join(', ');
  document.getElementById('phase12_tfs').value = (d.phase12_timeframes || []).join(', ');
  document.getElementById('phase3_tfs').value = (d.phase3_timeframes || []).join(', ');
  document.getElementById('seeds').value = (d.seeds || []).join(', ');
  document.getElementById('n_processes').value = d.n_processes ?? 1;
  document.getElementById('phase2_top_percent').value = d.phase2_top_percent ?? 10;
  document.getElementById('phase3_top_n').value = d.phase3_top_n ?? 30;
  document.getElementById('trials_text').value = d.trials_by_timeframe_text || '';
  document.getElementById('warn').textContent = DEF.warning || '';
  setMsg('');
  updateRunSummary();
}}

function payload() {{
  return {{
    strategies: [...document.querySelectorAll('.strategy:checked')].map(x => x.value),
    enable_phase1: document.getElementById('phase1').checked,
    enable_phase2: document.getElementById('phase2').checked,
    enable_phase3: document.getElementById('phase3').checked,
    pairs: document.getElementById('pairs').value,
    phase12_timeframes: document.getElementById('phase12_tfs').value,
    phase3_timeframes: document.getElementById('phase3_tfs').value,
    seeds: document.getElementById('seeds').value,
    n_processes: Number(document.getElementById('n_processes').value),
    phase2_top_percent: Number(document.getElementById('phase2_top_percent').value),
    phase3_top_n: Number(document.getElementById('phase3_top_n').value),
    trials_by_timeframe_text: document.getElementById('trials_text').value
  }};
}}

async function startRun() {{
  setMsg('Starting...');
  try {{
    const r = await fetch('/api/jobs/start', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload())
    }});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Failed');
    setMsg('Started ' + data.job.job_id);
    refreshState();
  }} catch (e) {{
    setMsg(e.message || String(e), true);
  }}
}}

async function stopRun() {{
  try {{
    const s = await (await fetch('/api/state')).json();
    if (!s.active_job) {{
      setMsg('No active job', true);
      return;
    }}
    const r = await fetch('/api/jobs/' + s.active_job.job_id + '/stop', {{ method: 'POST' }});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Failed');
    setMsg('Stop requested for ' + s.active_job.job_id);
    refreshState();
  }} catch (e) {{
    setMsg(e.message || String(e), true);
  }}
}}

async function resumeJob(jobId) {{
  setMsg('Resuming ' + jobId + '...');
  try {{
    const r = await fetch('/api/jobs/' + jobId + '/resume', {{ method: 'POST' }});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Failed');
    setMsg('Resumed as ' + data.job.job_id);
    refreshState();
  }} catch (e) {{
    setMsg(e.message || String(e), true);
  }}
}}

function renderHistory(jobs) {{
  const body = document.getElementById('history');
  body.innerHTML = '';
  if (!jobs || !jobs.length) {{
    body.innerHTML = '<tr><td colspan="5">No jobs yet.</td></tr>';
    return;
  }}
  for (const j of jobs) {{
    const phases = ['phase1','phase2','phase3'].filter(k => j.config && j.config['enable_' + k]).join(', ');
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{j.job_id}}</td><td>${{j.status}}</td><td>${{j.pid ?? ''}}</td><td>${{phases}}</td><td>${{(j.config?.strategies || []).join(', ')}}</td>`;
    body.appendChild(tr);
  }}
}}

function renderResumable(jobs) {{
  const body = document.getElementById('resumable');
  body.innerHTML = '';
  const resumable = (jobs || []).filter(j => ['stopped', 'failed'].includes(j.status));
  if (!resumable.length) {{
    body.innerHTML = '<tr><td colspan="6">No resumable runs yet.</td></tr>';
    return;
  }}
  for (const j of resumable) {{
    const phases = ['phase1','phase2','phase3'].filter(k => j.config && j.config['enable_' + k]).join(', ');
    const strategies = (j.config?.strategies || []).join(', ');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{j.job_id}}</td>
      <td>${{phases || '-'}}</td>
      <td>${{strategies || '-'}}</td>
      <td>${{j.status}}</td>
      <td>${{j.ended_at || ''}}</td>
      <td><button onclick="resumeJob('${{j.job_id}}')">Resume</button></td>
    `;
    body.appendChild(tr);
  }}
}}

function renderActive(j) {{
  const meta = document.getElementById('activeMeta');
  const logs = document.getElementById('logs');
  if (!j) {{
    meta.textContent = '';
    logs.textContent = 'No active job.';
    return;
  }}
  meta.textContent = `(${{j.job_id}} | ${{j.status}} | pid=${{j.pid ?? '-'}})`;
  logs.textContent = (j.log_tail || []).join('\\n') || 'Waiting for logs...';
  logs.scrollTop = logs.scrollHeight;
}}

async function refreshState() {{
  try {{
    const r = await fetch('/api/state');
    const data = await r.json();
    renderActive(data.active_job);
    renderHistory(data.jobs || []);
    renderResumable(data.jobs || []);
  }} catch (e) {{
    setMsg('Refresh failed: ' + (e.message || e), true);
  }}
}}

resetForm();
for (const id of ['phase1', 'phase2', 'phase3']) {{
  document.getElementById(id).addEventListener('change', updateRunSummary);
}}
for (const id of ['pairs', 'phase12_tfs', 'phase3_tfs', 'seeds', 'n_processes', 'phase2_top_percent', 'phase3_top_n', 'trials_text']) {{
  document.getElementById(id).addEventListener('input', updateRunSummary);
}}
refreshState();
timer = setInterval(refreshState, 2000);
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/api/state")
async def api_state() -> JSONResponse:
    active = JOBS.active()
    return JSONResponse(
        {
            "server_time": now_iso(),
            "active_job": ({**active.summary(), "log_tail": active.tail()} if active else None),
            "jobs": JOBS.list_summaries(),
        }
    )


@app.post("/api/jobs/start")
async def api_start(request: Request) -> JSONResponse:
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    try:
        config = normalize_start_payload(raw)
        job = JOBS.start(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start: {exc}") from exc
    return JSONResponse({"ok": True, "job": job.summary()})


@app.post("/api/jobs/{job_id}/stop")
async def api_stop(job_id: str) -> JSONResponse:
    try:
        job = JOBS.stop(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop: {exc}") from exc
    return JSONResponse({"ok": True, "job": job.summary()})


@app.post("/api/jobs/{job_id}/resume")
async def api_resume(job_id: str) -> JSONResponse:
    try:
        source = JOBS.get(job_id)
        if source.status not in {"stopped", "failed"}:
            raise RuntimeError(
                f"Job {job_id} status is '{source.status}'. Only stopped/failed jobs can be resumed."
            )
        config = json.loads(json.dumps(source.config))
        config["created_at"] = now_iso()
        job = JOBS.start(config)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to resume: {exc}") from exc
    return JSONResponse({"ok": True, "job": job.summary()})


def run_main_cycle(config_path: Path) -> int:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    print("[UI WORKER] Starting main.py backtesting cycle", flush=True)
    print(f"[UI WORKER] config={config_path}", flush=True)
    print(json.dumps(cfg, indent=2), flush=True)

    import main as main_module
    from scripts.config import PHASE2_TOP_PERCENT

    main_module.STRATEGY_KEYS_TO_RUN = list(cfg["strategies"])
    main_module.N_PROCESSES = int(cfg["n_processes"])
    main_module.clear_error_json_log()

    pairs = list(cfg["pairs"])
    seeds = list(cfg["seeds"])
    tfs12 = list(cfg["phase12_timeframes"])
    tfs3 = list(cfg["phase3_timeframes"])
    trials = dict(cfg["trials_by_timeframe"])
    top_percent = float(cfg.get("phase2_top_percent", PHASE2_TOP_PERCENT))
    top_n = int(cfg.get("phase3_top_n", 30))

    phases: list[dict[str, Any]] = []
    if cfg.get("enable_phase1"):
        phases.append(
            {
                "name": "phase1",
                "exploration_space": "default",
                "seeds": seeds,
                "timeframes": tfs12,
                "trials_by_timeframe": trials,
            }
        )
    if cfg.get("enable_phase2"):
        phases.append(
            {
                "name": "phase2",
                "exploration_space": f"top_{num_label(top_percent)}percent_parameterized",
                "seeds": seeds,
                "timeframes": tfs12,
                "trials_by_timeframe": trials,
                "top_percent": top_percent,
            }
        )
    if cfg.get("enable_phase3"):
        phase3 = {"name": "phase3", "top_n": top_n}
        if tfs3:
            phase3["timeframes"] = tfs3
        phases.append(phase3)
    if not phases:
        raise ValueError("No phases enabled")

    print(f"[UI WORKER] strategies={main_module.STRATEGY_KEYS_TO_RUN}", flush=True)
    print(f"[UI WORKER] n_processes={main_module.N_PROCESSES}", flush=True)
    print(f"[UI WORKER] pairs={len(pairs)}", flush=True)
    print(f"[UI WORKER] phases={[p['name'] for p in phases]}", flush=True)

    for phase in phases:
        name = phase["name"]
        print(f"[UI WORKER] Running {name}...", flush=True)
        if name in ("phase1", "phase2"):
            kwargs = {
                "pairs": pairs,
                "timeframes": phase["timeframes"],
                "seeds": phase["seeds"],
                "trials_by_timeframe": phase["trials_by_timeframe"],
                "exploration_space": phase["exploration_space"],
                "phase_name": name,
            }
            if name == "phase1":
                study_args = main_module.build_study_args_phase1(**kwargs)
            else:
                kwargs["top_percent"] = phase["top_percent"]
                study_args = main_module.build_study_args_phase2(**kwargs)
            print(f"[UI WORKER] {name} jobs={len(study_args)}", flush=True)
            main_module.run_all_studies(study_args, max_parallel_studies=main_module.N_PROCESSES)
        elif name == "phase3":
            main_module.run_phase3(phase)
        else:
            raise ValueError(f"Unknown phase: {name}")
        print(f"[UI WORKER] {name} complete", flush=True)
        time.sleep(1)

    final_best = main_module.BacktestSQLHelper(
        read_only=True
    ).get_best_strategy_per_pair_with_metrics()
    (ROOT / "best_strategies.json").write_text(
        json.dumps(final_best, indent=2), encoding="utf-8"
    )
    print("[UI WORKER] Wrote best_strategies.json", flush=True)
    return 0


def worker_main(config: str) -> int:
    try:
        return run_main_cycle(Path(config))
    except KeyboardInterrupt:
        print("[UI WORKER] Interrupted", flush=True)
        return 130
    except Exception as exc:
        print(f"[UI WORKER] Fatal: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 1


def serve_main(host: str, port: int) -> int:
    try:
        import uvicorn
    except Exception as exc:
        print("Install dependencies: pip install fastapi uvicorn", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    print(f"Open http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, reload=False, log_level="info")
    return 0


def cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FastAPI UI for main.py backtesting")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Start UI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    worker = sub.add_parser("worker", help="Internal worker mode")
    worker.add_argument("--config", required=True)

    parser.set_defaults(cmd="serve")
    args = parser.parse_args(argv)
    if args.cmd == "worker":
        return worker_main(args.config)
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    return serve_main(host, port)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(cli())
