from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sqlite3
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
JOBS_INDEX_PATH = JOBS_ROOT / "jobs_index.json"
LOG_TAIL_LINES = 120
LOG_MEMORY_LINES = 1200
LOG_MAX_LINE_CHARS = 320
TRIAL_LOG_SAMPLE_EVERY = 25
FULL_LOG_DEFAULT_LINES = 3000
FULL_LOG_MAX_LINES = 3000
DB_DEFAULT_PAGE_SIZE = 50
DB_MAX_PAGE_SIZE = 500
TRIAL_LINE_RE = re.compile(r"\bTrial\s+\d+\b")
PHASE_RUNNING_RE = re.compile(r"\[UI WORKER\] Running (phase\d)\.\.\.")
PHASE_COMPLETE_RE = re.compile(r"\[UI WORKER\] (phase\d) complete")
PHASE_JOBS_RE = re.compile(r"\[UI WORKER\] (phase\d) jobs=(\d+)")
STUDY_NAME_EXISTING_RE = re.compile(r"study with name '([^']+)'", re.IGNORECASE)
STUDY_NAME_CREATED_RE = re.compile(r"name:\s*([^\s]+)")


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


def tail_file_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    max_lines = min(int(max_lines), FULL_LOG_MAX_LINES)
    chunk_size = 8192
    data = bytearray()
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            pos = file_size
            line_count = 0
            while pos > 0 and line_count <= max_lines:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                line_count += chunk.count(b"\n")
                data[:0] = chunk
    except Exception:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-max_lines:]


def clamp_page_values(page: int, page_size: int) -> tuple[int, int]:
    p = max(int(page or 1), 1)
    s = max(1, min(int(page_size or DB_DEFAULT_PAGE_SIZE), DB_MAX_PAGE_SIZE))
    return p, s


def sqlite_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def parse_json_maybe(text: Any, fallback: Any) -> Any:
    try:
        if text is None:
            return fallback
        return json.loads(text)
    except Exception:
        return fallback


def fetchone_dict(cursor: sqlite3.Cursor) -> dict[str, Any]:
    row = cursor.fetchone()
    if row is None:
        return {}
    cols = [d[0] for d in cursor.description] if cursor.description else []
    return dict(zip(cols, row))


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


def phase_from_study_name(study_name: Optional[str]) -> Optional[str]:
    if not study_name:
        return None
    for phase in ("phase1", "phase2", "phase3"):
        if str(study_name).startswith(f"{phase}_"):
            return phase
    return None


def get_backtest_sql(read_only: bool = True):
    from scripts.data.sql import BacktestSQLHelper

    return BacktestSQLHelper(read_only=read_only)


def safe_sort_clause(
    requested: Optional[str],
    direction: Optional[str],
    allow_map: dict[str, str],
    default_key: str,
) -> tuple[str, str]:
    key = str(requested or default_key)
    column = allow_map.get(key, allow_map[default_key])
    dir_sql = "ASC" if str(direction or "desc").lower() == "asc" else "DESC"
    return column, dir_sql


def fetchall_dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cursor.description] if cursor.description else []
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def enabled_phase_names(config: dict[str, Any]) -> list[str]:
    phases: list[str] = []
    if config.get("enable_phase1"):
        phases.append("phase1")
    if config.get("enable_phase2"):
        phases.append("phase2")
    if config.get("enable_phase3"):
        phases.append("phase3")
    return phases


def estimate_phase12_trial_budget(config: dict[str, Any]) -> int:
    if not config.get("enable_phase1") and not config.get("enable_phase2"):
        return 0
    strategies = len(config.get("strategies", []))
    pairs = len(config.get("pairs", []))
    seeds = len(config.get("seeds", []))
    tfs = list(config.get("phase12_timeframes", []))
    trials = dict(config.get("trials_by_timeframe", {}))
    per_seed_pair = sum(int(trials.get(tf, 0)) for tf in tfs)
    base = max(strategies, 1) * max(pairs, 1) * max(seeds, 1) * max(per_seed_pair, 0)
    total = 0
    if config.get("enable_phase1"):
        total += base
    if config.get("enable_phase2"):
        total += base
    return total


def estimate_phase12_trial_budget_per_phase(config: dict[str, Any]) -> int:
    strategies = len(config.get("strategies", []))
    pairs = len(config.get("pairs", []))
    seeds = len(config.get("seeds", []))
    tfs = list(config.get("phase12_timeframes", []))
    trials = dict(config.get("trials_by_timeframe", {}))
    per_seed_pair = sum(int(trials.get(tf, 0)) for tf in tfs)
    return max(strategies, 1) * max(pairs, 1) * max(seeds, 1) * max(per_seed_pair, 0)


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
    attempts: int = 1
    stop_requested: bool = False
    enabled_phases: list[str] = field(default_factory=list)
    current_phase: Optional[str] = None
    completed_phases: list[str] = field(default_factory=list)
    phase_jobs: dict[str, int] = field(default_factory=dict)
    trial_budget_est: int = 0
    phase12_trials_per_phase_est: int = 0
    trials_finished: int = 0
    last_stop_note: Optional[str] = None
    suppressed_trial_logs: int = 0
    study_names: list[str] = field(default_factory=list)
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_MEMORY_LINES), repr=False)

    def remaining_trials_est(self) -> int:
        return max(int(self.trial_budget_est) - int(self.trials_finished), 0)

    def progress_note(self) -> str:
        if self.status in {"stopped", "failed"} and self.last_stop_note:
            return self.last_stop_note
        if self.current_phase:
            return f"Running {self.current_phase}"
        if self.completed_phases:
            return "Completed phases: " + ", ".join(self.completed_phases)
        return "Not started"

    def build_stop_note(self) -> str:
        if self.current_phase:
            stage = f"Stopped during {self.current_phase}"
        else:
            pending = [p for p in self.enabled_phases if p not in self.completed_phases]
            if pending:
                stage = f"Stopped before {pending[0]}"
            elif self.completed_phases:
                stage = f"Stopped after {self.completed_phases[-1]}"
            else:
                stage = "Stopped before phase execution"
        done = ", ".join(self.completed_phases) if self.completed_phases else "none"
        return (
            f"{stage}; completed phases={done}; "
            f"est trials remaining={self.remaining_trials_est():,}"
        )

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
            "attempts": self.attempts,
            "stop_requested": self.stop_requested,
            "progress": {
                "current_phase": self.current_phase,
                "completed_phases": list(self.completed_phases),
                "trial_budget_est": self.trial_budget_est,
                "phase12_trials_per_phase_est": self.phase12_trials_per_phase_est,
                "trials_finished": self.trials_finished,
                "trial_remaining_est": self.remaining_trials_est(),
                "phase_jobs": dict(self.phase_jobs),
                "note": self.progress_note(),
                "study_names_count": len(self.study_names),
            },
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
        self._last_persist_monotonic = 0.0
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()
        self._recover_jobs_from_disk()
        self._persist_state(force=True)

    def _persist_state(self, force: bool = False) -> None:
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_persist_monotonic) < 2.0:
            return
        with self.lock:
            payload = {
                "saved_at": now_iso(),
                "order": list(self.order),
                "jobs": [self._job_record(self.jobs[jid]) for jid in self.order],
            }
        try:
            tmp = JOBS_INDEX_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(JOBS_INDEX_PATH)
            self._last_persist_monotonic = now_mono
        except Exception:
            pass

    @staticmethod
    def _job_record(job: Job) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "config": job.config,
            "config_path": job.config_path,
            "log_path": job.log_path,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "ended_at": job.ended_at,
            "pid": job.pid,
            "return_code": job.return_code,
            "error": job.error,
            "attempts": job.attempts,
            "stop_requested": job.stop_requested,
            "enabled_phases": list(job.enabled_phases),
            "current_phase": job.current_phase,
            "completed_phases": list(job.completed_phases),
            "phase_jobs": dict(job.phase_jobs),
            "trial_budget_est": job.trial_budget_est,
            "phase12_trials_per_phase_est": job.phase12_trials_per_phase_est,
            "trials_finished": job.trials_finished,
            "last_stop_note": job.last_stop_note,
            "study_names": list(job.study_names),
        }

    @staticmethod
    def _job_from_record(data: dict[str, Any]) -> Job:
        job = Job(
            job_id=str(data.get("job_id", "")),
            config=dict(data.get("config", {})),
            config_path=str(data.get("config_path", "")),
            log_path=str(data.get("log_path", "")),
            status=str(data.get("status", "stopped")),
            created_at=str(data.get("created_at", now_iso())),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            pid=data.get("pid"),
            return_code=data.get("return_code"),
            error=data.get("error"),
            attempts=int(data.get("attempts", 1)),
            stop_requested=bool(data.get("stop_requested", False)),
            enabled_phases=list(data.get("enabled_phases", [])),
            current_phase=data.get("current_phase"),
            completed_phases=list(data.get("completed_phases", [])),
            phase_jobs=dict(data.get("phase_jobs", {})),
            trial_budget_est=int(data.get("trial_budget_est", 0)),
            phase12_trials_per_phase_est=int(data.get("phase12_trials_per_phase_est", 0)),
            trials_finished=int(data.get("trials_finished", 0)),
            last_stop_note=data.get("last_stop_note"),
            study_names=list(data.get("study_names", [])),
        )
        if job.status in {"running", "stopping"}:
            job.status = "stopped"
            job.stop_requested = True
            job.ended_at = job.ended_at or now_iso()
            if not job.last_stop_note:
                job.last_stop_note = "Recovered after UI restart while job was running."
        if not job.enabled_phases:
            job.enabled_phases = enabled_phase_names(job.config)
        if job.trial_budget_est <= 0:
            job.trial_budget_est = estimate_phase12_trial_budget(job.config)
        if job.phase12_trials_per_phase_est <= 0:
            job.phase12_trials_per_phase_est = estimate_phase12_trial_budget_per_phase(job.config)
        return job

    def _load_persisted_jobs(self) -> None:
        if not JOBS_INDEX_PATH.exists():
            return
        try:
            payload = json.loads(JOBS_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        jobs_raw = payload.get("jobs")
        if not isinstance(jobs_raw, list):
            return

        by_id: dict[str, Job] = {}
        for item in jobs_raw:
            if not isinstance(item, dict):
                continue
            try:
                job = self._job_from_record(item)
            except Exception:
                continue
            if job.job_id:
                by_id[job.job_id] = job

        order_raw = payload.get("order")
        order: list[str] = []
        if isinstance(order_raw, list):
            for jid in order_raw:
                jid = str(jid)
                if jid in by_id and jid not in order:
                    order.append(jid)
        for jid in sorted(by_id.keys()):
            if jid not in order:
                order.append(jid)

        self.jobs = {jid: by_id[jid] for jid in order}
        self.order = order

    def _recover_jobs_from_disk(self) -> None:
        try:
            job_dirs = [p for p in JOBS_ROOT.iterdir() if p.is_dir()]
        except Exception:
            return
        for job_dir in sorted(job_dirs):
            jid = job_dir.name
            if jid in self.jobs:
                continue
            cfg_path = job_dir / "config.json"
            log_path = job_dir / "run.log"
            if not cfg_path.exists():
                continue
            try:
                config = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            ended = None
            if log_path.exists():
                try:
                    ended = datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds")
                except Exception:
                    ended = now_iso()
            else:
                ended = now_iso()
            job = Job(
                job_id=jid,
                config=config,
                config_path=str(cfg_path),
                log_path=str(log_path),
                status="stopped",
                created_at=now_iso(),
                started_at=None,
                ended_at=ended,
                attempts=1,
                stop_requested=True,
            )
            self._reset_job_progress(job)
            job.status = "stopped"
            job.stop_requested = True
            job.ended_at = ended
            job.last_stop_note = "Recovered from disk after UI restart."
            self.jobs[jid] = job
            self.order.append(jid)

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

    def _spawn_worker_process(self, config_path: Path) -> subprocess.Popen:
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return subprocess.Popen(
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

    def _reset_job_progress(self, job: Job) -> None:
        job.enabled_phases = enabled_phase_names(job.config)
        job.current_phase = None
        job.completed_phases.clear()
        job.phase_jobs.clear()
        job.trials_finished = 0
        job.trial_budget_est = estimate_phase12_trial_budget(job.config)
        job.phase12_trials_per_phase_est = estimate_phase12_trial_budget_per_phase(job.config)
        job.last_stop_note = None
        job.suppressed_trial_logs = 0

    def _update_progress_from_log(self, job: Job, line: str) -> None:
        text = line.strip()
        if not text:
            return
        study_name = None
        existing_match = STUDY_NAME_EXISTING_RE.search(text)
        if existing_match:
            study_name = existing_match.group(1).strip()
        else:
            created_match = STUDY_NAME_CREATED_RE.search(text)
            if created_match:
                candidate = created_match.group(1).strip().rstrip(".;,")
                if candidate.startswith(("phase1_", "phase2_", "phase3_")):
                    study_name = candidate
        if study_name and study_name not in job.study_names:
            job.study_names.append(study_name)

        match = PHASE_RUNNING_RE.search(text)
        if match:
            job.current_phase = match.group(1)

        match = PHASE_COMPLETE_RE.search(text)
        if match:
            phase = match.group(1)
            if phase not in job.completed_phases:
                job.completed_phases.append(phase)
            if job.current_phase == phase:
                job.current_phase = None
            if phase in {"phase1", "phase2"} and job.phase12_trials_per_phase_est > 0:
                completed_phase12 = len([p for p in job.completed_phases if p in {"phase1", "phase2"}])
                min_finished = completed_phase12 * job.phase12_trials_per_phase_est
                if job.trials_finished < min_finished:
                    job.trials_finished = min_finished

        match = PHASE_JOBS_RE.search(text)
        if match:
            job.phase_jobs[match.group(1)] = int(match.group(2))

        lowered = text.lower()
        if "trial" in lowered and ("finished" in lowered or "pruned" in lowered):
            if TRIAL_LINE_RE.search(text):
                job.trials_finished += 1

    @staticmethod
    def _trim_ui_log_line(text: str) -> str:
        if len(text) <= LOG_MAX_LINE_CHARS:
            return text
        return text[: LOG_MAX_LINE_CHARS - 3] + "..."

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
            proc = self._spawn_worker_process(config_path)

            job = Job(
                job_id=job_id,
                config=config,
                config_path=str(config_path),
                log_path=str(log_path),
                pid=proc.pid,
                process=proc,
            )
            self._reset_job_progress(job)
            self.jobs[job_id] = job
            self.order.append(job_id)
            threading.Thread(target=self._read_logs, args=(job_id,), daemon=True).start()
            threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()
        self._persist_state(force=True)
        return job

    def resume(self, job_id: str) -> Job:
        with self.lock:
            active = self.active()
            if active:
                raise RuntimeError(f"Job {active.job_id} is still {active.status}")
            job = self.get(job_id)
            if job.status not in {"stopped", "failed"}:
                raise RuntimeError(
                    f"Job {job_id} status is '{job.status}'. Only stopped/failed jobs can be resumed."
                )
            config_path = Path(job.config_path)
            if not config_path.exists():
                raise RuntimeError(f"Missing config file for {job_id}: {config_path}")

            proc = self._spawn_worker_process(config_path)
            job.attempts += 1
            job.status = "running"
            job.started_at = now_iso()
            job.ended_at = None
            job.pid = proc.pid
            job.return_code = None
            job.error = None
            job.stop_requested = False
            job.process = proc
            job.logs.clear()
            self._reset_job_progress(job)
            marker = f"[UI] Resume attempt {job.attempts} started at {job.started_at}"
            job.logs.append(marker)
            if job_id in self.order:
                self.order.remove(job_id)
            self.order.append(job_id)
            log_path = Path(job.log_path)

        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n" + marker + "\n")
        except Exception:
            pass

        threading.Thread(target=self._read_logs, args=(job_id,), daemon=True).start()
        threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()
        self._persist_state(force=True)
        return job

    def _append_log(self, job_id: str, line: str) -> None:
        should_persist = False
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            self._update_progress_from_log(job, line)
            text = line.rstrip("\n")
            lowered = text.lower()
            is_trial_detail = bool(
                TRIAL_LINE_RE.search(text)
                and ("finished" in lowered or "pruned" in lowered)
            )
            if is_trial_detail:
                if job.trials_finished % TRIAL_LOG_SAMPLE_EVERY != 0:
                    job.suppressed_trial_logs += 1
                    return
                if job.suppressed_trial_logs:
                    job.logs.append(
                        f"[UI] Suppressed {job.suppressed_trial_logs:,} trial log lines for responsiveness."
                    )
                    job.suppressed_trial_logs = 0
            elif job.suppressed_trial_logs:
                job.logs.append(
                    f"[UI] Suppressed {job.suppressed_trial_logs:,} trial log lines for responsiveness."
                )
                job.suppressed_trial_logs = 0
            job.logs.append(self._trim_ui_log_line(text))
            should_persist = True
        if should_persist:
            self._persist_state(force=False)

    def _read_logs(self, job_id: str) -> None:
        try:
            job = self.get(job_id)
            proc = job.process
            if proc is None or proc.stdout is None:
                return
            log_path = Path(job.log_path)
            with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    try:
                        log_file.write(line)
                    except Exception:
                        pass
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
            should_persist = False
            with self.lock:
                j = self.jobs.get(job_id)
                if not j:
                    return
                j.return_code = rc
                j.ended_at = now_iso()
                if j.stop_requested:
                    j.status = "stopped"
                    j.last_stop_note = j.build_stop_note()
                else:
                    j.status = "completed" if rc == 0 else "failed"
                    if j.status == "failed":
                        j.last_stop_note = j.build_stop_note()
                    else:
                        j.current_phase = None
                should_persist = True
            if should_persist:
                self._persist_state(force=True)
        except Exception as exc:
            should_persist = False
            with self.lock:
                j = self.jobs.get(job_id)
                if j:
                    j.status = "failed"
                    j.error = f"watcher error: {exc}"
                    j.ended_at = now_iso()
                    j.last_stop_note = j.build_stop_note()
                    should_persist = True
            if should_persist:
                self._persist_state(force=True)

    def stop(self, job_id: str) -> Job:
        with self.lock:
            job = self.get(job_id)
            proc = job.process
            if proc is None or proc.poll() is not None:
                return job
            job.stop_requested = True
            job.status = "stopping"
            pid = proc.pid
        self._persist_state(force=True)

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
        job = self.get(job_id)
        self._persist_state(force=True)
        return job


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
    html, body {{ height:100%; }}
    body {{
      font-family: Segoe UI, sans-serif;
      margin: 10px;
      overflow: hidden;
      display:flex;
      flex-direction:column;
      gap:6px;
    }}
    h2 {{ margin: 0 0 6px 0; font-size: 22px; }}
    .mode-tabs {{ display:flex; gap:8px; flex-wrap:wrap; }}
    .mode-btn {{
      border:1px solid #b8c3cf;
      background:#f4f8fc;
      color:#31465c;
      border-radius:999px;
      font-size:12px;
      padding:5px 12px;
      cursor:pointer;
    }}
    .mode-btn.active {{ background:#dfefff; border-color:#8bb3dd; font-weight:600; }}
    #msg {{ min-height:16px; font-size:12px; }}
    .wrap {{
      display:grid;
      grid-template-columns: minmax(0, 520px) minmax(0, 1fr);
      gap: 10px;
      align-items: stretch;
      flex:1;
      min-height:0;
    }}
    .box {{ border:1px solid #ccc; border-radius:8px; padding:10px; min-width:0; }}
    .control-box {{ display:flex; flex-direction:column; min-height:0; overflow:hidden; }}
    .right-col {{ display:grid; gap:10px; min-height:0; overflow:auto; padding-right:2px; }}
    label {{ display:block; margin-top:5px; font-size:11.5px; color:#444; }}
    input, textarea {{ width:100%; box-sizing:border-box; padding:5px; font-size:12px; }}
    textarea {{ min-height:52px; }}
    .hint {{ margin-top:4px; font-size:11px; color:#666; line-height:1.4; }}
    .phase-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:8px; }}
    #strategies {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:2px 10px; }}
    .setup-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:8px 10px; align-items:start; }}
    .setup-col {{ min-width:0; }}
    .btns {{ margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; }}
    .log-toolbar {{ margin: 6px 0 4px 0; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    .log-toolbar label {{ margin:0; font-size:11px; color:#445; display:flex; gap:4px; align-items:center; }}
    .log-toolbar select {{ width:auto; padding:3px 4px; font-size:11px; }}
    pre {{ background:#111; color:#fff; padding:8px; min-height:180px; max-height:300px; overflow:auto; }}
    pre.log-pre {{
      white-space: pre-wrap;
      word-break: break-word;
      overflow-x: hidden;
      color:#fff;
      font-size:11px;
      line-height:1.3;
    }}
    .log-info {{ color:#39ff14; font-weight:600; }}
    pre.summary-pre {{
      background:#f7f9fc;
      color:#263242;
      border:1px solid #d7dee8;
      min-height:0;
      max-height:none;
      margin-top:6px;
      margin-bottom:0;
      font-size:11px;
      line-height:1.35;
      max-height:84px;
      overflow:auto;
    }}
    .metrics {{ margin-top:6px; }}
    .tabs {{ display:flex; gap:6px; margin-top:6px; margin-bottom:6px; }}
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
    #tab_setup.active {{ height:100%; overflow:hidden; }}
    #pairs {{ min-height:52px; max-height:52px; }}
    #trials_text {{ min-height:96px; max-height:96px; }}
    .guide-box {{
      border:1px solid #d7dee8;
      border-radius:8px;
      padding:8px 10px;
      background:#fbfdff;
    }}
    .guide-item {{ margin-top:6px; font-size:12px; color:#2f3d4e; line-height:1.4; }}
    table {{ width:100%; border-collapse: collapse; font-size:12px; table-layout: fixed; }}
    th, td {{ border-bottom:1px solid #ddd; padding:4px; text-align:left; word-break: break-word; }}
    .db-tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
    .db-tab-btn {{
      border:1px solid #ccd8e4;
      background:#f6f9fc;
      color:#1f3d57;
      border-radius:8px;
      padding:4px 8px;
      font-size:11px;
      cursor:pointer;
    }}
    .db-tab-btn.active {{ background:#dff0ff; border-color:#8eb8dc; font-weight:600; }}
    .db-tab-panel {{ display:none; margin-top:8px; }}
    .db-tab-panel.active {{ display:block; }}
    .db-filters {{
      display:grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap:6px;
      margin-bottom:8px;
    }}
    .db-filters .full {{ grid-column: 1 / -1; }}
    .db-actions {{ display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin:6px 0; }}
    .db-actions select, .db-actions input {{ width:auto; max-width:260px; }}
    .db-scroll {{ max-height:260px; overflow:auto; border:1px solid #d9e0e7; border-radius:6px; }}
    .db-pager {{ display:flex; gap:6px; align-items:center; margin-top:6px; font-size:11px; color:#445; }}
    .db-note {{ font-size:11px; color:#567; }}
    .mono {{ font-family: Consolas, "Courier New", monospace; }}
    pre.db-pre {{
      background:#0f1319;
      color:#e9eef3;
      border:1px solid #2a3340;
      min-height:90px;
      max-height:200px;
    }}
    body.backtesting-mode #db_console {{ display:none; }}
    body.database-mode .wrap {{ grid-template-columns: minmax(0, 1fr); }}
    body.database-mode .control-box {{ display:none; }}
    body.database-mode .right-col {{ overflow:hidden; padding-right:0; }}
    body.database-mode .right-col > :not(#db_console) {{ display:none; }}
    body.database-mode #db_console {{
      display:block;
      margin:0;
      height:100%;
      overflow:auto;
    }}
    body.database-mode .db-scroll {{ max-height:52vh; }}
    @media (max-width: 1400px) {{ .db-filters {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 1400px) {{ .wrap {{ grid-template-columns: minmax(0, 480px) minmax(0, 1fr); }} }}
    @media (max-width: 1200px) {{
      body {{ overflow:auto; }}
      .wrap {{ height:auto; grid-template-columns: 1fr; }}
      .control-box {{ overflow:visible; }}
      .right-col {{ overflow:visible; }}
      .setup-grid {{ grid-template-columns: 1fr; }}
      .phase-grid {{ grid-template-columns: 1fr; }}
      #strategies {{ grid-template-columns: 1fr; }}
      .db-filters {{ grid-template-columns: 1fr; }}
      #pairs, #trials_text {{ max-height:none; }}
    }}
  </style>
</head>
<body class="backtesting-mode">
  <h2>main.py Backtesting UI</h2>
  <div class="mode-tabs">
    <button id="mode_backtesting_btn" class="mode-btn active" onclick="switchAppMode('backtesting')">Backtesting</button>
    <button id="mode_database_btn" class="mode-btn" onclick="switchAppMode('database')">Database</button>
  </div>
  <div id="warn" style="color:#8a5a00"></div>
  <div id="msg"></div>
  <div class="wrap">
    <div class="box control-box">
      <div><b>Backtest Controls</b></div>
      <div class="hint">Use tabs to switch between setup and guidance without long scrolling.</div>
      <div class="tabs">
        <button id="tab_setup_btn" class="tab-btn active" onclick="switchTab('setup')">Setup</button>
        <button id="tab_guide_btn" class="tab-btn" onclick="switchTab('guide')">Guide</button>
      </div>

      <div id="tab_setup" class="tab-panel active">
        <div class="setup-grid">
          <div class="setup-col">
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
          </div>

          <div class="setup-col">
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
            <textarea id="trials_text"></textarea>
            <div class="hint">Optimization attempts per timeframe in Phases 1/2. Higher values increase runtime.</div>
            <div class="metrics">
              <div><b>Run Size Estimate</b></div>
              <pre id="run_summary" class="summary-pre"></pre>
            </div>
          </div>
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
    </div>
    <div class="right-col">
      <div class="box">
        <div><b>Active Job</b> <span id="activeMeta"></span></div>
        <div class="log-toolbar">
          <label><input type="checkbox" id="full_logs_mode" checked> Full log tail (all lines)</label>
          <label>Lines
            <select id="full_logs_lines">
              <option value="300">300</option>
              <option value="800">800</option>
              <option value="1500">1500</option>
              <option value="3000" selected>3000 (max)</option>
            </select>
          </label>
        </div>
        <div class="hint">Default is compact live logs for speed. Full mode reads unsampled lines from disk tail.</div>
        <pre id="logs" class="log-pre">No active job.</pre>
      </div>
      <div class="box">
        <div><b>Resumable Runs</b> <span class="hint">(stopped or failed)</span></div>
        <table>
          <thead><tr><th>Job</th><th>Attempts</th><th>Progress</th><th>Trials Left (est)</th><th>Status</th><th>Resume</th></tr></thead>
          <tbody id="resumable"></tbody>
        </table>
      </div>
      <div class="box">
        <div><b>History</b></div>
        <table><thead><tr><th>Job</th><th>Attempts</th><th>Status</th><th>PID</th><th>Phases</th><th>Strategies</th></tr></thead><tbody id="history"></tbody></table>
      </div>
      <div class="box" id="db_console">
        <div><b>Database Console</b> <span class="db-note">Phase 1 tools</span></div>
        <div class="db-actions">
          <label class="db-note">Saved View</label>
          <select id="db_saved_views"></select>
          <button onclick="saveCurrentDbView()">Save</button>
          <button onclick="loadCurrentDbView()">Load</button>
          <button onclick="deleteCurrentDbView()">Delete</button>
          <button onclick="exportCurrentDbTab('json')">Export JSON</button>
          <button onclick="exportCurrentDbTab('csv')">Export CSV</button>
        </div>
        <div class="db-tabs">
          <button id="db_btn_health" class="db-tab-btn active" onclick="switchDbTab('health')">Health</button>
          <button id="db_btn_tables" class="db-tab-btn" onclick="switchDbTab('tables')">Tables</button>
          <button id="db_btn_runs" class="db-tab-btn" onclick="switchDbTab('runs')">Runs</button>
          <button id="db_btn_studies" class="db-tab-btn" onclick="switchDbTab('studies')">Studies</button>
          <button id="db_btn_scores" class="db-tab-btn" onclick="switchDbTab('scores')">Scores</button>
          <button id="db_btn_trades" class="db-tab-btn" onclick="switchDbTab('trades')">Trades</button>
          <button id="db_btn_top" class="db-tab-btn" onclick="switchDbTab('top')">Top Strats</button>
          <button id="db_btn_trace" class="db-tab-btn" onclick="switchDbTab('trace')">Job Trace</button>
        </div>

        <div id="db_tab_health" class="db-tab-panel active">
          <div class="db-actions">
            <button onclick="loadDbHealth()">Refresh Health</button>
          </div>
          <pre id="db_health_out" class="db-pre mono">Loading database health...</pre>
        </div>

        <div id="db_tab_tables" class="db-tab-panel">
          <div class="db-filters">
            <div>
              <label>Table</label>
              <select id="db_table_name"></select>
            </div>
            <div>
              <label>Page Size</label>
              <input id="db_table_page_size" type="number" value="50" min="1" max="500">
            </div>
            <div>
              <label>Sort By</label>
              <input id="db_table_sort_by" placeholder="column">
            </div>
            <div>
              <label>Sort Dir</label>
              <select id="db_table_sort_dir"><option value="asc">asc</option><option value="desc">desc</option></select>
            </div>
          </div>
          <div class="db-actions">
            <button onclick="loadDbTable(true)">Load Table</button>
          </div>
          <div id="db_table_meta" class="db-note"></div>
          <div class="db-scroll">
            <table>
              <thead id="db_table_head"></thead>
              <tbody id="db_table_body"></tbody>
            </table>
          </div>
          <div class="db-pager">
            <button onclick="changeDbTablePage(-1)">Prev</button>
            <span id="db_table_pager">page 1</span>
            <button onclick="changeDbTablePage(1)">Next</button>
          </div>
        </div>

        <div id="db_tab_runs" class="db-tab-panel">
          <div class="db-filters">
            <div><label>Pair</label><input id="db_runs_pair" placeholder="EUR/USD"></div>
            <div><label>Timeframe</label><input id="db_runs_timeframe" placeholder="2_hour"></div>
            <div><label>Strategy Name</label><input id="db_runs_strategy_name"></div>
            <div><label>Phase</label><select id="db_runs_phase"><option value="">all</option><option value="phase1">phase1</option><option value="phase2">phase2</option><option value="phase3">phase3</option></select></div>
            <div><label>Date From</label><input id="db_runs_date_from" placeholder="YYYY-MM-DD"></div>
            <div><label>Date To</label><input id="db_runs_date_to" placeholder="YYYY-MM-DD"></div>
            <div><label>Score Min</label><input id="db_runs_score_min" type="number" step="0.01"></div>
            <div><label>Score Max</label><input id="db_runs_score_max" type="number" step="0.01"></div>
          </div>
          <div class="db-actions">
            <button onclick="loadDbRuns(true)">Apply Filters</button>
            <label class="db-note">Page Size</label><input id="db_runs_page_size" type="number" value="50" min="1" max="500">
          </div>
          <div class="db-scroll">
            <table>
              <thead><tr><th>ID</th><th>Pair</th><th>TF</th><th>Strategy</th><th>Score</th><th>Phase</th><th>Net</th><th>Return%</th><th>Actions</th></tr></thead>
              <tbody id="db_runs_body"></tbody>
            </table>
          </div>
          <div class="db-pager">
            <button onclick="changeDbRunsPage(-1)">Prev</button>
            <span id="db_runs_pager">page 1</span>
            <button onclick="changeDbRunsPage(1)">Next</button>
          </div>
        </div>

        <div id="db_tab_studies" class="db-tab-panel">
          <div class="db-filters">
            <div><label>Pair</label><input id="db_studies_pair"></div>
            <div><label>Timeframe</label><input id="db_studies_timeframe"></div>
            <div><label>Exploration Space</label><input id="db_studies_space"></div>
            <div><label>Stop Reason</label><input id="db_studies_stop_reason"></div>
            <div><label>Study Name</label><input id="db_studies_name"></div>
            <div><label>Min Trials</label><input id="db_studies_min_trials" type="number"></div>
            <div><label>Max Trials</label><input id="db_studies_max_trials" type="number"></div>
            <div><label>Min Best Score</label><input id="db_studies_min_score" type="number" step="0.01"></div>
          </div>
          <div class="db-actions">
            <button onclick="loadDbStudies(true)">Apply Filters</button>
            <label class="db-note">Page Size</label><input id="db_studies_page_size" type="number" value="50" min="1" max="500">
          </div>
          <div class="db-scroll">
            <table>
              <thead><tr><th>ID</th><th>Study</th><th>Pair</th><th>TF</th><th>Space</th><th>Best</th><th>Trials</th><th>Stop</th></tr></thead>
              <tbody id="db_studies_body"></tbody>
            </table>
          </div>
          <div class="db-pager">
            <button onclick="changeDbStudiesPage(-1)">Prev</button>
            <span id="db_studies_pager">page 1</span>
            <button onclick="changeDbStudiesPage(1)">Next</button>
          </div>
        </div>

        <div id="db_tab_scores" class="db-tab-panel">
          <div class="db-filters">
            <div><label>Pair</label><input id="db_scores_pair"></div>
            <div><label>Timeframe</label><input id="db_scores_timeframe"></div>
            <div><label>Study Name</label><input id="db_scores_study"></div>
            <div><label>Space</label><input id="db_scores_space"></div>
            <div><label>Score Min</label><input id="db_scores_min" type="number" step="0.01"></div>
            <div><label>Score Max</label><input id="db_scores_max" type="number" step="0.01"></div>
            <div><label>Sort By</label><select id="db_scores_sort_by"><option value="score">score</option><option value="timestamp">timestamp</option><option value="pair">pair</option><option value="timeframe">timeframe</option></select></div>
            <div><label>Sort Dir</label><select id="db_scores_sort_dir"><option value="desc">desc</option><option value="asc">asc</option></select></div>
          </div>
          <div class="db-actions">
            <button onclick="loadDbScores(true)">Apply Filters</button>
            <label class="db-note">Page Size</label><input id="db_scores_page_size" type="number" value="50" min="1" max="500">
          </div>
          <div class="db-scroll">
            <table>
              <thead><tr><th>ID</th><th>Score</th><th>Study</th><th>Trial</th><th>Pair</th><th>TF</th><th>Strategy</th><th>Actions</th></tr></thead>
              <tbody id="db_scores_body"></tbody>
            </table>
          </div>
          <div class="db-pager">
            <button onclick="changeDbScoresPage(-1)">Prev</button>
            <span id="db_scores_pager">page 1</span>
            <button onclick="changeDbScoresPage(1)">Next</button>
          </div>
        </div>

        <div id="db_tab_trades" class="db-tab-panel">
          <div class="db-filters">
            <div><label>Backtest ID</label><input id="db_trades_backtest_id" type="number"></div>
            <div><label>Sort By</label><select id="db_trades_sort_by"><option value="timestamp">timestamp</option><option value="pnl">pnl</option><option value="return_pct">return_pct</option><option value="duration">duration</option></select></div>
            <div><label>Sort Dir</label><select id="db_trades_sort_dir"><option value="desc">desc</option><option value="asc">asc</option></select></div>
            <div><label>Page Size</label><input id="db_trades_page_size" type="number" value="100" min="1" max="500"></div>
          </div>
          <div class="db-actions">
            <button onclick="loadDbTrades(true)">Load Trades</button>
            <span id="db_trades_summary" class="db-note"></span>
          </div>
          <div class="db-scroll">
            <table>
              <thead><tr><th>ID</th><th>Entry</th><th>Exit</th><th>Dir</th><th>PnL</th><th>Ret%</th><th>Pips</th><th>Dur</th></tr></thead>
              <tbody id="db_trades_body"></tbody>
            </table>
          </div>
          <div class="db-pager">
            <button onclick="changeDbTradesPage(-1)">Prev</button>
            <span id="db_trades_pager">page 1</span>
            <button onclick="changeDbTradesPage(1)">Next</button>
          </div>
        </div>

        <div id="db_tab_top" class="db-tab-panel">
          <div class="db-filters">
            <div><label>Mode</label><select id="db_top_mode"><option value="percent">top percent</option><option value="top_n">top N</option><option value="best_pair">best per pair</option></select></div>
            <div><label>Exploration Space</label><input id="db_top_space" value="default"></div>
            <div><label>Top Percent</label><input id="db_top_percent" type="number" value="10" step="0.1"></div>
            <div><label>Top N</label><input id="db_top_n" type="number" value="25"></div>
            <div class="full"><label>Exclude Spaces (comma)</label><input id="db_top_exclude" value="generalization_test"></div>
          </div>
          <div class="db-actions">
            <button onclick="loadDbTopStrategies()">Load Top Strategies</button>
          </div>
          <div class="db-scroll">
            <table>
              <thead><tr><th>Score</th><th>Pair</th><th>TF</th><th>Strategy ID</th><th>Study</th><th>Actions</th></tr></thead>
              <tbody id="db_top_body"></tbody>
            </table>
          </div>
        </div>

        <div id="db_tab_trace" class="db-tab-panel">
          <div class="db-actions">
            <label class="db-note">Job</label>
            <select id="db_trace_job_id"></select>
            <button onclick="loadDbTrace()">Load Trace</button>
          </div>
          <div class="db-note mono" id="db_trace_study_names"></div>
          <div class="db-scroll">
            <table>
              <thead><tr><th>Study</th><th>Pair</th><th>TF</th><th>Space</th><th>Best</th><th>Trials</th></tr></thead>
              <tbody id="db_trace_studies_body"></tbody>
            </table>
          </div>
          <div class="db-scroll" style="margin-top:8px;">
            <table>
              <thead><tr><th>Backtest</th><th>Study</th><th>Pair</th><th>TF</th><th>Strategy</th><th>Best Score</th></tr></thead>
              <tbody id="db_trace_backtests_body"></tbody>
            </table>
          </div>
        </div>

        <div style="margin-top:10px;">
          <div><b>Strategy Detail</b></div>
          <pre id="db_strategy_detail" class="db-pre mono">Select a strategy from Runs, Scores, or Top Strats.</pre>
          <div class="db-scroll">
            <table>
              <thead><tr><th>Backtest</th><th>Pair</th><th>TF</th><th>Score</th><th>Net</th><th>Actions</th></tr></thead>
              <tbody id="db_strategy_runs_body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
<script>
const DEF = {defaults_json};
let timer = null;
let fullLogFetchErrorShown = false;
const DB_SAVED_VIEWS_KEY = 'backtest_ui_db_saved_views_v1';
const DB_STATE = {{
  activeTab: 'health',
  schema: null,
  table: {{ page: 1, page_size: 50, total_pages: 1, rows: [] }},
  runs: {{ page: 1, page_size: 50, total_pages: 1, rows: [] }},
  studies: {{ page: 1, page_size: 50, total_pages: 1, rows: [] }},
  scores: {{ page: 1, page_size: 50, total_pages: 1, rows: [] }},
  trades: {{ page: 1, page_size: 100, total_pages: 1, rows: [], run: null }},
  top: {{ rows: [] }},
  trace: {{ studies: [], backtests: [] }},
}};
const UI_STATE = {{
  mode: 'backtesting',
  dbInitialized: false,
}};

function setMsg(text, err=false) {{
  const el = document.getElementById('msg');
  el.textContent = text || '';
  el.style.color = err ? '#b00020' : '#1d5d28';
}}

function toNumberOrNull(v) {{
  if (v === '' || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}}

function q(params) {{
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {{}})) {{
    if (v === null || v === undefined || v === '') continue;
    sp.set(k, String(v));
  }}
  return sp.toString();
}}

async function dbFetch(url) {{
  const r = await fetch(url);
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Database request failed');
  return data;
}}

function downloadText(filename, text, mime='text/plain;charset=utf-8') {{
  const blob = new Blob([text], {{ type: mime }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}

function rowsToCsv(rows) {{
  if (!rows || !rows.length) return '';
  const cols = [...new Set(rows.flatMap(r => Object.keys(r || {{}})))];
  const esc = (x) => {{
    const s = String(x ?? '');
    if (/[\",\\n]/.test(s)) return '"' + s.replaceAll('"', '""') + '"';
    return s;
  }};
  const lines = [];
  lines.push(cols.map(esc).join(','));
  for (const row of rows) {{
    lines.push(cols.map(c => esc(row[c])).join(','));
  }}
  return lines.join('\\n');
}}

function fullLogsEnabled() {{
  const el = document.getElementById('full_logs_mode');
  return !!(el && el.checked);
}}

function fullLogsLines() {{
  const el = document.getElementById('full_logs_lines');
  const n = Number((el && el.value) || {FULL_LOG_DEFAULT_LINES});
  return Number.isFinite(n) && n > 0 ? n : {FULL_LOG_DEFAULT_LINES};
}}

function escapeHtml(text) {{
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

function highlightLogLine(line) {{
  const escaped = escapeHtml(line);
  return escaped.replace(/^(\\[[^\\]]+\\])/, '<span class="log-info">$1</span>');
}}

function switchAppMode(mode) {{
  const dbMode = mode === 'database';
  UI_STATE.mode = dbMode ? 'database' : 'backtesting';
  document.body.classList.toggle('database-mode', dbMode);
  document.body.classList.toggle('backtesting-mode', !dbMode);
  document.getElementById('mode_backtesting_btn').classList.toggle('active', !dbMode);
  document.getElementById('mode_database_btn').classList.toggle('active', dbMode);

  if (dbMode) {{
    refreshDbSavedViewsSelect();
    if (!UI_STATE.dbInitialized) {{
      switchDbTab(DB_STATE.activeTab || 'health');
      loadDbSchema();
      UI_STATE.dbInitialized = true;
    }}
  }}
}}

function switchTab(name) {{
  const setup = name === 'setup';
  document.getElementById('tab_setup').classList.toggle('active', setup);
  document.getElementById('tab_guide').classList.toggle('active', !setup);
  document.getElementById('tab_setup_btn').classList.toggle('active', setup);
  document.getElementById('tab_guide_btn').classList.toggle('active', !setup);
}}

function switchDbTab(tab) {{
  DB_STATE.activeTab = tab;
  for (const btn of document.querySelectorAll('.db-tab-btn')) {{
    btn.classList.remove('active');
  }}
  const activeBtn = document.getElementById('db_btn_' + tab);
  if (activeBtn) activeBtn.classList.add('active');
  for (const panel of document.querySelectorAll('.db-tab-panel')) {{
    panel.classList.remove('active');
  }}
  const activePanel = document.getElementById('db_tab_' + tab);
  if (activePanel) activePanel.classList.add('active');
  refreshDbSavedViewsSelect();
  if (tab === 'health') loadDbHealth();
  if (tab === 'tables') loadDbSchema().then(() => loadDbTable(false));
  if (tab === 'runs') loadDbRuns(false);
  if (tab === 'studies') loadDbStudies(false);
  if (tab === 'scores') loadDbScores(false);
  if (tab === 'trades') loadDbTrades(false);
  if (tab === 'top') loadDbTopStrategies();
  if (tab === 'trace') loadDbTrace();
}}

function getDbSavedViews() {{
  try {{
    const raw = localStorage.getItem(DB_SAVED_VIEWS_KEY);
    const data = raw ? JSON.parse(raw) : {{}};
    return (data && typeof data === 'object') ? data : {{}};
  }} catch (_) {{
    return {{}};
  }}
}}

function setDbSavedViews(data) {{
  localStorage.setItem(DB_SAVED_VIEWS_KEY, JSON.stringify(data || {{}}));
}}

function getCurrentDbFilters(tab) {{
  if (tab === 'runs') {{
    return {{
      pair: document.getElementById('db_runs_pair').value,
      timeframe: document.getElementById('db_runs_timeframe').value,
      strategy_name: document.getElementById('db_runs_strategy_name').value,
      phase: document.getElementById('db_runs_phase').value,
      date_from: document.getElementById('db_runs_date_from').value,
      date_to: document.getElementById('db_runs_date_to').value,
      score_min: document.getElementById('db_runs_score_min').value,
      score_max: document.getElementById('db_runs_score_max').value,
      page_size: document.getElementById('db_runs_page_size').value,
    }};
  }}
  if (tab === 'studies') {{
    return {{
      pair: document.getElementById('db_studies_pair').value,
      timeframe: document.getElementById('db_studies_timeframe').value,
      space: document.getElementById('db_studies_space').value,
      stop_reason: document.getElementById('db_studies_stop_reason').value,
      name: document.getElementById('db_studies_name').value,
      min_trials: document.getElementById('db_studies_min_trials').value,
      max_trials: document.getElementById('db_studies_max_trials').value,
      min_score: document.getElementById('db_studies_min_score').value,
      page_size: document.getElementById('db_studies_page_size').value,
    }};
  }}
  if (tab === 'scores') {{
    return {{
      pair: document.getElementById('db_scores_pair').value,
      timeframe: document.getElementById('db_scores_timeframe').value,
      study: document.getElementById('db_scores_study').value,
      space: document.getElementById('db_scores_space').value,
      min: document.getElementById('db_scores_min').value,
      max: document.getElementById('db_scores_max').value,
      sort_by: document.getElementById('db_scores_sort_by').value,
      sort_dir: document.getElementById('db_scores_sort_dir').value,
      page_size: document.getElementById('db_scores_page_size').value,
    }};
  }}
  if (tab === 'trades') {{
    return {{
      backtest_id: document.getElementById('db_trades_backtest_id').value,
      sort_by: document.getElementById('db_trades_sort_by').value,
      sort_dir: document.getElementById('db_trades_sort_dir').value,
      page_size: document.getElementById('db_trades_page_size').value,
    }};
  }}
  if (tab === 'top') {{
    return {{
      mode: document.getElementById('db_top_mode').value,
      space: document.getElementById('db_top_space').value,
      top_percent: document.getElementById('db_top_percent').value,
      top_n: document.getElementById('db_top_n').value,
      exclude: document.getElementById('db_top_exclude').value,
    }};
  }}
  if (tab === 'tables') {{
    return {{
      table: document.getElementById('db_table_name').value,
      page_size: document.getElementById('db_table_page_size').value,
      sort_by: document.getElementById('db_table_sort_by').value,
      sort_dir: document.getElementById('db_table_sort_dir').value,
    }};
  }}
  return {{}};
}}

function applyDbFilters(tab, data) {{
  const d = data || {{}};
  if (tab === 'runs') {{
    document.getElementById('db_runs_pair').value = d.pair || '';
    document.getElementById('db_runs_timeframe').value = d.timeframe || '';
    document.getElementById('db_runs_strategy_name').value = d.strategy_name || '';
    document.getElementById('db_runs_phase').value = d.phase || '';
    document.getElementById('db_runs_date_from').value = d.date_from || '';
    document.getElementById('db_runs_date_to').value = d.date_to || '';
    document.getElementById('db_runs_score_min').value = d.score_min || '';
    document.getElementById('db_runs_score_max').value = d.score_max || '';
    document.getElementById('db_runs_page_size').value = d.page_size || '50';
    DB_STATE.runs.page = 1;
  }} else if (tab === 'studies') {{
    document.getElementById('db_studies_pair').value = d.pair || '';
    document.getElementById('db_studies_timeframe').value = d.timeframe || '';
    document.getElementById('db_studies_space').value = d.space || '';
    document.getElementById('db_studies_stop_reason').value = d.stop_reason || '';
    document.getElementById('db_studies_name').value = d.name || '';
    document.getElementById('db_studies_min_trials').value = d.min_trials || '';
    document.getElementById('db_studies_max_trials').value = d.max_trials || '';
    document.getElementById('db_studies_min_score').value = d.min_score || '';
    document.getElementById('db_studies_page_size').value = d.page_size || '50';
    DB_STATE.studies.page = 1;
  }} else if (tab === 'scores') {{
    document.getElementById('db_scores_pair').value = d.pair || '';
    document.getElementById('db_scores_timeframe').value = d.timeframe || '';
    document.getElementById('db_scores_study').value = d.study || '';
    document.getElementById('db_scores_space').value = d.space || '';
    document.getElementById('db_scores_min').value = d.min || '';
    document.getElementById('db_scores_max').value = d.max || '';
    document.getElementById('db_scores_sort_by').value = d.sort_by || 'score';
    document.getElementById('db_scores_sort_dir').value = d.sort_dir || 'desc';
    document.getElementById('db_scores_page_size').value = d.page_size || '50';
    DB_STATE.scores.page = 1;
  }} else if (tab === 'trades') {{
    document.getElementById('db_trades_backtest_id').value = d.backtest_id || '';
    document.getElementById('db_trades_sort_by').value = d.sort_by || 'timestamp';
    document.getElementById('db_trades_sort_dir').value = d.sort_dir || 'desc';
    document.getElementById('db_trades_page_size').value = d.page_size || '100';
    DB_STATE.trades.page = 1;
  }} else if (tab === 'top') {{
    document.getElementById('db_top_mode').value = d.mode || 'percent';
    document.getElementById('db_top_space').value = d.space || 'default';
    document.getElementById('db_top_percent').value = d.top_percent || '10';
    document.getElementById('db_top_n').value = d.top_n || '25';
    document.getElementById('db_top_exclude').value = d.exclude || 'generalization_test';
  }} else if (tab === 'tables') {{
    if (d.table) document.getElementById('db_table_name').value = d.table;
    document.getElementById('db_table_page_size').value = d.page_size || '50';
    document.getElementById('db_table_sort_by').value = d.sort_by || '';
    document.getElementById('db_table_sort_dir').value = d.sort_dir || 'asc';
    DB_STATE.table.page = 1;
  }}
}}

function refreshDbSavedViewsSelect() {{
  const sel = document.getElementById('db_saved_views');
  if (!sel) return;
  const store = getDbSavedViews();
  const tab = DB_STATE.activeTab;
  const views = (store[tab] && typeof store[tab] === 'object') ? store[tab] : {{}};
  const names = Object.keys(views).sort();
  sel.innerHTML = '';
  const empty = document.createElement('option');
  empty.value = '';
  empty.textContent = '(select)';
  sel.appendChild(empty);
  for (const n of names) {{
    const opt = document.createElement('option');
    opt.value = n;
    opt.textContent = n;
    sel.appendChild(opt);
  }}
}}

function saveCurrentDbView() {{
  const tab = DB_STATE.activeTab;
  const name = prompt(`Save view name for ${{tab}}:`);
  if (!name) return;
  const key = name.trim();
  if (!key) return;
  const store = getDbSavedViews();
  if (!store[tab] || typeof store[tab] !== 'object') store[tab] = {{}};
  store[tab][key] = getCurrentDbFilters(tab);
  setDbSavedViews(store);
  refreshDbSavedViewsSelect();
  document.getElementById('db_saved_views').value = key;
  setMsg(`Saved DB view '${{key}}' for ${{tab}}.`);
}}

function loadCurrentDbView() {{
  const tab = DB_STATE.activeTab;
  const sel = document.getElementById('db_saved_views');
  const key = sel.value;
  if (!key) return;
  const store = getDbSavedViews();
  const data = store?.[tab]?.[key];
  if (!data) {{
    setMsg('Saved view not found', true);
    return;
  }}
  applyDbFilters(tab, data);
  if (tab === 'runs') loadDbRuns(true);
  if (tab === 'studies') loadDbStudies(true);
  if (tab === 'scores') loadDbScores(true);
  if (tab === 'trades') loadDbTrades(true);
  if (tab === 'top') loadDbTopStrategies();
  if (tab === 'tables') loadDbTable(true);
  setMsg(`Loaded DB view '${{key}}' for ${{tab}}.`);
}}

function deleteCurrentDbView() {{
  const tab = DB_STATE.activeTab;
  const sel = document.getElementById('db_saved_views');
  const key = sel.value;
  if (!key) return;
  const store = getDbSavedViews();
  if (store?.[tab]?.[key] === undefined) return;
  delete store[tab][key];
  setDbSavedViews(store);
  refreshDbSavedViewsSelect();
  setMsg(`Deleted DB view '${{key}}' for ${{tab}}.`);
}}

function exportCurrentDbTab(fmt) {{
  const tab = DB_STATE.activeTab;
  const rows = (
    tab === 'runs' ? DB_STATE.runs.rows :
    tab === 'studies' ? DB_STATE.studies.rows :
    tab === 'scores' ? DB_STATE.scores.rows :
    tab === 'trades' ? DB_STATE.trades.rows :
    tab === 'top' ? DB_STATE.top.rows :
    tab === 'trace' ? DB_STATE.trace.backtests :
    tab === 'tables' ? DB_STATE.table.rows : []
  ) || [];
  if (!rows.length) {{
    setMsg('No rows to export on this tab.', true);
    return;
  }}
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  if (fmt === 'json') {{
    downloadText(`db_${{tab}}_${{stamp}}.json`, JSON.stringify(rows, null, 2), 'application/json;charset=utf-8');
  }} else {{
    downloadText(`db_${{tab}}_${{stamp}}.csv`, rowsToCsv(rows), 'text/csv;charset=utf-8');
  }}
}}

async function loadDbHealth() {{
  try {{
    const data = await dbFetch('/api/db/health');
    const lines = [];
    lines.push(`DB: ${{data.db_path}}`);
    lines.push(`Exists: ${{data.exists}}`);
    lines.push(`Size (MB): ${{((data.size_bytes || 0) / (1024 * 1024)).toFixed(2)}}`);
    lines.push(`Modified: ${{data.modified_at || '-'}}`);
    lines.push(`Journal: ${{data.journal_mode}} | pages: ${{data.page_count}} x ${{data.page_size}}`);
    lines.push('');
    lines.push('Table counts:');
    for (const [k, v] of Object.entries(data.table_counts || {{}})) {{
      lines.push(`- ${{k}}: ${{v}}`);
    }}
    document.getElementById('db_health_out').textContent = lines.join('\\n');
  }} catch (e) {{
    document.getElementById('db_health_out').textContent = String(e?.message || e);
  }}
}}

async function loadDbSchema() {{
  try {{
    const data = await dbFetch('/api/db/schema');
    DB_STATE.schema = data;
    const sel = document.getElementById('db_table_name');
    const current = sel.value;
    sel.innerHTML = '';
    for (const t of (data.tables || [])) {{
      const opt = document.createElement('option');
      opt.value = t.name;
      opt.textContent = `${{t.name}} (${{t.row_count}})`;
      sel.appendChild(opt);
    }}
    if (current && [...sel.options].some(o => o.value === current)) {{
      sel.value = current;
    }} else if (data.default_table) {{
      sel.value = data.default_table;
    }}
  }} catch (e) {{
    setMsg('Schema load failed: ' + (e.message || e), true);
  }}
}}

async function loadDbTable(resetPage=false) {{
  try {{
    if (resetPage) DB_STATE.table.page = 1;
    const table = document.getElementById('db_table_name').value;
    const pageSize = Number(document.getElementById('db_table_page_size').value || 50);
    DB_STATE.table.page_size = pageSize;
    const sortBy = document.getElementById('db_table_sort_by').value.trim();
    const sortDir = document.getElementById('db_table_sort_dir').value;
    const qs = q({{ table, page: DB_STATE.table.page, page_size: pageSize, sort_by: sortBy, sort_dir: sortDir }});
    const data = await dbFetch('/api/db/table?' + qs);
    DB_STATE.table.rows = data.rows || [];
    DB_STATE.table.total_pages = data.total_pages || 1;
    const head = document.getElementById('db_table_head');
    head.innerHTML = '<tr>' + (data.columns || []).map(c => `<th>${{escapeHtml(c)}}</th>`).join('') + '</tr>';
    const body = document.getElementById('db_table_body');
    body.innerHTML = '';
    for (const row of DB_STATE.table.rows) {{
      const tr = document.createElement('tr');
      tr.innerHTML = (data.columns || []).map(c => `<td>${{escapeHtml(row[c])}}</td>`).join('');
      body.appendChild(tr);
    }}
    document.getElementById('db_table_meta').textContent =
      `Rows: ${{data.total_rows}} | page ${{data.page}} / ${{Math.max(data.total_pages, 1)}}`;
    document.getElementById('db_table_pager').textContent =
      `page ${{data.page}} / ${{Math.max(data.total_pages, 1)}}`;
  }} catch (e) {{
    setMsg('Table load failed: ' + (e.message || e), true);
  }}
}}

function changeDbTablePage(delta) {{
  DB_STATE.table.page = Math.max(1, DB_STATE.table.page + delta);
  loadDbTable(false);
}}

async function loadDbRuns(resetPage=false) {{
  try {{
    if (resetPage) DB_STATE.runs.page = 1;
    DB_STATE.runs.page_size = Number(document.getElementById('db_runs_page_size').value || 50);
    const qs = q({{
      page: DB_STATE.runs.page,
      page_size: DB_STATE.runs.page_size,
      pair: document.getElementById('db_runs_pair').value.trim(),
      timeframe: document.getElementById('db_runs_timeframe').value.trim(),
      strategy_name: document.getElementById('db_runs_strategy_name').value.trim(),
      phase: document.getElementById('db_runs_phase').value,
      date_from: document.getElementById('db_runs_date_from').value.trim(),
      date_to: document.getElementById('db_runs_date_to').value.trim(),
      score_min: toNumberOrNull(document.getElementById('db_runs_score_min').value),
      score_max: toNumberOrNull(document.getElementById('db_runs_score_max').value),
    }});
    const data = await dbFetch('/api/db/backtest_runs?' + qs);
    DB_STATE.runs.rows = data.rows || [];
    DB_STATE.runs.total_pages = data.total_pages || 1;
    const body = document.getElementById('db_runs_body');
    body.innerHTML = '';
    for (const r of DB_STATE.runs.rows) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{r.backtest_id}}</td>
        <td>${{escapeHtml(r.pair)}}</td>
        <td>${{escapeHtml(r.timeframe)}}</td>
        <td><button onclick="showStrategyDetail(${{r.strategy_id}})">${{escapeHtml(r.strategy_name)}}</button></td>
        <td>${{r.score ?? ''}}</td>
        <td>${{escapeHtml(r.phase || '')}}</td>
        <td>${{r.net_profit ?? ''}}</td>
        <td>${{r.total_return_pct ?? ''}}</td>
        <td>
          <button onclick="openTrades(${{r.backtest_id}})">Trades</button>
          <button onclick="resumeFromBacktest(${{r.backtest_id}})">Resume</button>
        </td>
      `;
      body.appendChild(tr);
    }}
    document.getElementById('db_runs_pager').textContent =
      `page ${{data.page}} / ${{Math.max(data.total_pages, 1)}} | total ${{data.total_rows}}`;
  }} catch (e) {{
    setMsg('Runs load failed: ' + (e.message || e), true);
  }}
}}

function changeDbRunsPage(delta) {{
  DB_STATE.runs.page = Math.max(1, DB_STATE.runs.page + delta);
  loadDbRuns(false);
}}

async function loadDbStudies(resetPage=false) {{
  try {{
    if (resetPage) DB_STATE.studies.page = 1;
    DB_STATE.studies.page_size = Number(document.getElementById('db_studies_page_size').value || 50);
    const qs = q({{
      page: DB_STATE.studies.page,
      page_size: DB_STATE.studies.page_size,
      pair: document.getElementById('db_studies_pair').value.trim(),
      timeframe: document.getElementById('db_studies_timeframe').value.trim(),
      exploration_space: document.getElementById('db_studies_space').value.trim(),
      stop_reason: document.getElementById('db_studies_stop_reason').value.trim(),
      study_name: document.getElementById('db_studies_name').value.trim(),
      min_trials: toNumberOrNull(document.getElementById('db_studies_min_trials').value),
      max_trials: toNumberOrNull(document.getElementById('db_studies_max_trials').value),
      min_best_score: toNumberOrNull(document.getElementById('db_studies_min_score').value),
    }});
    const data = await dbFetch('/api/db/studies?' + qs);
    DB_STATE.studies.rows = data.rows || [];
    DB_STATE.studies.total_pages = data.total_pages || 1;
    const body = document.getElementById('db_studies_body');
    body.innerHTML = '';
    for (const r of DB_STATE.studies.rows) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{r.id}}</td>
        <td>${{escapeHtml(r.study_name)}}</td>
        <td>${{escapeHtml(r.pair)}}</td>
        <td>${{escapeHtml(r.timeframe)}}</td>
        <td>${{escapeHtml(r.exploration_space)}}</td>
        <td>${{r.best_score ?? ''}}</td>
        <td>${{r.n_trials ?? ''}}</td>
        <td>${{escapeHtml(r.stop_reason)}}</td>
      `;
      body.appendChild(tr);
    }}
    document.getElementById('db_studies_pager').textContent =
      `page ${{data.page}} / ${{Math.max(data.total_pages, 1)}} | total ${{data.total_rows}}`;
  }} catch (e) {{
    setMsg('Studies load failed: ' + (e.message || e), true);
  }}
}}

function changeDbStudiesPage(delta) {{
  DB_STATE.studies.page = Math.max(1, DB_STATE.studies.page + delta);
  loadDbStudies(false);
}}

async function loadDbScores(resetPage=false) {{
  try {{
    if (resetPage) DB_STATE.scores.page = 1;
    DB_STATE.scores.page_size = Number(document.getElementById('db_scores_page_size').value || 50);
    const qs = q({{
      page: DB_STATE.scores.page,
      page_size: DB_STATE.scores.page_size,
      pair: document.getElementById('db_scores_pair').value.trim(),
      timeframe: document.getElementById('db_scores_timeframe').value.trim(),
      study_name: document.getElementById('db_scores_study').value.trim(),
      exploration_space: document.getElementById('db_scores_space').value.trim(),
      score_min: toNumberOrNull(document.getElementById('db_scores_min').value),
      score_max: toNumberOrNull(document.getElementById('db_scores_max').value),
      sort_by: document.getElementById('db_scores_sort_by').value,
      sort_dir: document.getElementById('db_scores_sort_dir').value,
    }});
    const data = await dbFetch('/api/db/composite_scores?' + qs);
    DB_STATE.scores.rows = data.rows || [];
    DB_STATE.scores.total_pages = data.total_pages || 1;
    const body = document.getElementById('db_scores_body');
    body.innerHTML = '';
    for (const r of DB_STATE.scores.rows) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{r.composite_id}}</td>
        <td>${{r.score ?? ''}}</td>
        <td>${{escapeHtml(r.study_name)}}</td>
        <td>${{r.trial_id ?? ''}}</td>
        <td>${{escapeHtml(r.pair)}}</td>
        <td>${{escapeHtml(r.timeframe)}}</td>
        <td><button onclick="showStrategyDetail(${{r.strategy_id}})">${{escapeHtml(r.strategy_name || '')}}</button></td>
        <td>
          <button onclick="openTrades(${{r.backtest_id}})">Trades</button>
          <button onclick="resumeFromBacktest(${{r.backtest_id}})">Resume</button>
        </td>
      `;
      body.appendChild(tr);
    }}
    document.getElementById('db_scores_pager').textContent =
      `page ${{data.page}} / ${{Math.max(data.total_pages, 1)}} | total ${{data.total_rows}}`;
  }} catch (e) {{
    setMsg('Scores load failed: ' + (e.message || e), true);
  }}
}}

function changeDbScoresPage(delta) {{
  DB_STATE.scores.page = Math.max(1, DB_STATE.scores.page + delta);
  loadDbScores(false);
}}

async function openTrades(backtestId) {{
  document.getElementById('db_trades_backtest_id').value = backtestId || '';
  DB_STATE.trades.page = 1;
  switchAppMode('database');
  switchDbTab('trades');
  await loadDbTrades(true);
}}

async function loadDbTrades(resetPage=false) {{
  try {{
    const backtestId = Number(document.getElementById('db_trades_backtest_id').value || 0);
    if (!backtestId) {{
      if (resetPage) setMsg('Provide Backtest ID for trades.', true);
      return;
    }}
    if (resetPage) DB_STATE.trades.page = 1;
    DB_STATE.trades.page_size = Number(document.getElementById('db_trades_page_size').value || 100);
    const qs = q({{
      backtest_id: backtestId,
      page: DB_STATE.trades.page,
      page_size: DB_STATE.trades.page_size,
      sort_by: document.getElementById('db_trades_sort_by').value,
      sort_dir: document.getElementById('db_trades_sort_dir').value,
    }});
    const data = await dbFetch('/api/db/trades?' + qs);
    DB_STATE.trades.rows = data.rows || [];
    DB_STATE.trades.run = data.run || null;
    DB_STATE.trades.total_pages = data.total_pages || 1;
    const body = document.getElementById('db_trades_body');
    body.innerHTML = '';
    for (const t of DB_STATE.trades.rows) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{t.id}}</td>
        <td>${{escapeHtml(t.timestamp)}}</td>
        <td>${{escapeHtml(t.exit_timestamp)}}</td>
        <td>${{t.direction ?? ''}}</td>
        <td>${{t.pnl ?? ''}}</td>
        <td>${{t.return_pct ?? ''}}</td>
        <td>${{t.net_pips ?? ''}}</td>
        <td>${{t.duration_minutes ?? ''}}</td>
      `;
      body.appendChild(tr);
    }}
    const run = data.run || {{}};
    document.getElementById('db_trades_summary').textContent =
      `Run ${{run.backtest_id}} | ${{run.pair || ''}} ${{run.timeframe || ''}} | strategy=${{run.strategy_name || ''}} | rows=${{data.total_rows}}`;
    document.getElementById('db_trades_pager').textContent =
      `page ${{data.page}} / ${{Math.max(data.total_pages, 1)}}`;
  }} catch (e) {{
    setMsg('Trades load failed: ' + (e.message || e), true);
  }}
}}

function changeDbTradesPage(delta) {{
  DB_STATE.trades.page = Math.max(1, DB_STATE.trades.page + delta);
  loadDbTrades(false);
}}

async function showStrategyDetail(strategyId) {{
  try {{
    if (!strategyId) return;
    const data = await dbFetch('/api/db/strategy/' + strategyId);
    const s = data.strategy || {{}};
    document.getElementById('db_strategy_detail').textContent = JSON.stringify(s, null, 2);
    const body = document.getElementById('db_strategy_runs_body');
    body.innerHTML = '';
    for (const r of (data.linked_runs || [])) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{r.backtest_id}}</td>
        <td>${{escapeHtml(r.pair)}}</td>
        <td>${{escapeHtml(r.timeframe)}}</td>
        <td>${{r.score ?? ''}}</td>
        <td>${{r.net_profit ?? ''}}</td>
        <td>
          <button onclick="openTrades(${{r.backtest_id}})">Trades</button>
          <button onclick="resumeFromBacktest(${{r.backtest_id}})">Resume</button>
        </td>
      `;
      body.appendChild(tr);
    }}
  }} catch (e) {{
    setMsg('Strategy detail failed: ' + (e.message || e), true);
  }}
}}

async function resumeFromBacktest(backtestId) {{
  try {{
    const data = await dbFetch('/api/db/run_config/' + backtestId);
    const p = data.payload || null;
    if (!p) throw new Error('No reconstructed payload');
    setFormFromPayload(p);
    await startRunWithPayload(p);
  }} catch (e) {{
    setMsg('Resume config failed: ' + (e.message || e), true);
  }}
}}

async function loadDbTopStrategies() {{
  try {{
    const mode = document.getElementById('db_top_mode').value;
    const qs = q({{
      mode,
      exploration_space: document.getElementById('db_top_space').value.trim(),
      top_percent: toNumberOrNull(document.getElementById('db_top_percent').value),
      top_n: toNumberOrNull(document.getElementById('db_top_n').value),
      exclude_spaces: document.getElementById('db_top_exclude').value.trim(),
    }});
    const data = await dbFetch('/api/db/top_strategies?' + qs);
    DB_STATE.top.rows = data.rows || [];
    const body = document.getElementById('db_top_body');
    body.innerHTML = '';
    DB_STATE.top.rows.forEach((row, idx) => {{
      const score = row.Score ?? row.score ?? '';
      const pair = row.Pair ?? row.pair ?? '';
      const tf = row.Timeframe ?? row.timeframe ?? '';
      const sid = row.StrategyConfig_ID ?? row.strategy_id ?? '';
      const study = row.Study_Name ?? row.study_name ?? '';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{score}}</td>
        <td>${{escapeHtml(pair)}}</td>
        <td>${{escapeHtml(tf)}}</td>
        <td>${{sid}}</td>
        <td>${{escapeHtml(study)}}</td>
        <td>
          ${{sid ? `<button onclick="showStrategyDetail(${{sid}})">Detail</button>` : ''}}
          <button onclick="resumeFromTopRow(${{idx}})">Resume</button>
        </td>
      `;
      body.appendChild(tr);
    }});
  }} catch (e) {{
    setMsg('Top strategies load failed: ' + (e.message || e), true);
  }}
}}

async function resumeFromTopRow(index) {{
  try {{
    const row = DB_STATE.top.rows[index];
    if (!row) return;
    const pair = row.Pair ?? row.pair;
    const tf = row.Timeframe ?? row.timeframe;
    const sid = row.StrategyConfig_ID ?? row.strategy_id;
    if (!pair || !tf || !sid) {{
      throw new Error('Need pair/timeframe/strategy id on selected row.');
    }}
    const detail = await dbFetch('/api/db/strategy/' + sid + '?limit_runs=1');
    const strategyName = detail?.strategy?.name;
    if (!strategyName) throw new Error('Strategy name unavailable.');
    const d = DEF.default || {{}};
    const trials = {{ ...(d.trials_by_timeframe || {{}}) }};
    if (!(tf in trials)) trials[tf] = 200;
    const p = {{
      strategies: [strategyName],
      pairs: [pair],
      phase12_timeframes: [tf],
      phase3_timeframes: [],
      seeds: [...(d.seeds || [42])],
      n_processes: Number(d.n_processes ?? 1),
      phase2_top_percent: Number(d.phase2_top_percent ?? 10),
      phase3_top_n: Number(d.phase3_top_n ?? 30),
      enable_phase1: true,
      enable_phase2: false,
      enable_phase3: false,
      trials_by_timeframe_text: Object.entries(trials).map(([k, v]) => `${{k}}=${{v}}`).join('\\n')
    }};
    setFormFromPayload(p);
    await startRunWithPayload(p);
  }} catch (e) {{
    setMsg('Resume from top strategy failed: ' + (e.message || e), true);
  }}
}}

function refreshTraceJobOptions(jobs) {{
  const sel = document.getElementById('db_trace_job_id');
  if (!sel) return;
  const prev = sel.value;
  const items = (jobs || []).map(j => j.job_id);
  sel.innerHTML = items.map(id => `<option value="${{escapeHtml(id)}}">${{escapeHtml(id)}}</option>`).join('');
  if (prev && items.includes(prev)) sel.value = prev;
}}

async function loadDbTrace() {{
  try {{
    const jobId = document.getElementById('db_trace_job_id').value;
    if (!jobId) {{
      document.getElementById('db_trace_study_names').textContent = 'No job selected.';
      return;
    }}
    const data = await dbFetch('/api/db/trace/' + encodeURIComponent(jobId));
    DB_STATE.trace.studies = data.studies || [];
    DB_STATE.trace.backtests = data.backtests || [];
    document.getElementById('db_trace_study_names').textContent =
      'Study names: ' + ((data.study_names || []).join(', ') || '(none)');

    const studiesBody = document.getElementById('db_trace_studies_body');
    studiesBody.innerHTML = '';
    for (const s of DB_STATE.trace.studies) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{escapeHtml(s.study_name)}}</td>
        <td>${{escapeHtml(s.pair)}}</td>
        <td>${{escapeHtml(s.timeframe)}}</td>
        <td>${{escapeHtml(s.exploration_space)}}</td>
        <td>${{s.best_score ?? ''}}</td>
        <td>${{s.n_trials ?? ''}}</td>
      `;
      studiesBody.appendChild(tr);
    }}

    const btBody = document.getElementById('db_trace_backtests_body');
    btBody.innerHTML = '';
    for (const b of DB_STATE.trace.backtests) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{b.backtest_id}}</td>
        <td>${{escapeHtml(b.study_name)}}</td>
        <td>${{escapeHtml(b.pair)}}</td>
        <td>${{escapeHtml(b.timeframe)}}</td>
        <td><button onclick="showStrategyDetail(${{b.strategy_id}})">${{escapeHtml(b.strategy_name)}}</button></td>
        <td>${{b.best_score ?? ''}}</td>
      `;
      btBody.appendChild(tr);
    }}
  }} catch (e) {{
    setMsg('Trace load failed: ' + (e.message || e), true);
  }}
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

function setFormFromPayload(p) {{
  if (!p || typeof p !== 'object') return;
  for (const cb of document.querySelectorAll('.strategy')) {{
    cb.checked = Array.isArray(p.strategies) && p.strategies.includes(cb.value);
  }}
  if ('enable_phase1' in p) document.getElementById('phase1').checked = !!p.enable_phase1;
  if ('enable_phase2' in p) document.getElementById('phase2').checked = !!p.enable_phase2;
  if ('enable_phase3' in p) document.getElementById('phase3').checked = !!p.enable_phase3;
  if (Array.isArray(p.pairs)) document.getElementById('pairs').value = p.pairs.join(', ');
  if (Array.isArray(p.phase12_timeframes)) document.getElementById('phase12_tfs').value = p.phase12_timeframes.join(', ');
  if (Array.isArray(p.phase3_timeframes)) document.getElementById('phase3_tfs').value = p.phase3_timeframes.join(', ');
  if (Array.isArray(p.seeds)) document.getElementById('seeds').value = p.seeds.join(', ');
  if (typeof p.n_processes === 'number') document.getElementById('n_processes').value = p.n_processes;
  if (typeof p.phase2_top_percent === 'number') document.getElementById('phase2_top_percent').value = p.phase2_top_percent;
  if (typeof p.phase3_top_n === 'number') document.getElementById('phase3_top_n').value = p.phase3_top_n;
  if (typeof p.trials_by_timeframe_text === 'string') {{
    document.getElementById('trials_text').value = p.trials_by_timeframe_text;
  }} else if (p.trials_by_timeframe && typeof p.trials_by_timeframe === 'object') {{
    const lines = Object.entries(p.trials_by_timeframe).map(([k, v]) => `${{k}}=${{v}}`);
    document.getElementById('trials_text').value = lines.join('\\n');
  }}
  updateRunSummary();
}}

async function startRunWithPayload(runPayload) {{
  setMsg('Starting...');
  try {{
    const r = await fetch('/api/jobs/start', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(runPayload)
    }});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Failed');
    setMsg('Started ' + data.job.job_id);
    refreshState();
  }} catch (e) {{
    setMsg(e.message || String(e), true);
  }}
}}

async function startRun() {{
  await startRunWithPayload(payload());
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
    setMsg('Resumed ' + data.job.job_id + ' (attempt ' + (data.job.attempts || 1) + ')');
    refreshState();
  }} catch (e) {{
    setMsg(e.message || String(e), true);
  }}
}}

function renderHistory(jobs) {{
  const body = document.getElementById('history');
  body.innerHTML = '';
  if (!jobs || !jobs.length) {{
    body.innerHTML = '<tr><td colspan="6">No jobs yet.</td></tr>';
    return;
  }}
  for (const j of jobs) {{
    const phases = ['phase1','phase2','phase3'].filter(k => j.config && j.config['enable_' + k]).join(', ');
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{j.job_id}}</td><td>${{j.attempts || 1}}</td><td>${{j.status}}</td><td>${{j.pid ?? ''}}</td><td>${{phases}}</td><td>${{(j.config?.strategies || []).join(', ')}}</td>`;
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
    const progress = j.progress || {{}};
    const note = progress.note || '-';
    const remaining = Number.isFinite(progress.trial_remaining_est)
      ? Number(progress.trial_remaining_est).toLocaleString()
      : '-';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{j.job_id}}</td>
      <td>${{j.attempts || 1}}</td>
      <td>${{note}}</td>
      <td>${{remaining}}</td>
      <td>${{j.status}}</td>
      <td><button onclick="resumeJob('${{j.job_id}}')">Resume</button></td>
    `;
    body.appendChild(tr);
  }}
}}

function renderActive(j, linesOverride=null, modeLabel='') {{
  const meta = document.getElementById('activeMeta');
  const logs = document.getElementById('logs');
  if (!j) {{
    meta.textContent = '';
    logs.textContent = 'No active job.';
    return;
  }}
  const modeSuffix = modeLabel ? ` | ${{modeLabel}}` : '';
  meta.textContent = `(${{j.job_id}} | ${{j.status}} | pid=${{j.pid ?? '-'}}${{modeSuffix}})`;
  const lines = Array.isArray(linesOverride) ? linesOverride : (j.log_tail || []);
  if (!lines.length) {{
    logs.textContent = 'Waiting for logs...';
  }} else {{
    logs.innerHTML = lines.map(highlightLogLine).join('<br>');
  }}
  logs.scrollTop = logs.scrollHeight;
}}

async function fetchFullLogs(jobId) {{
  const lines = fullLogsLines();
  const r = await fetch(`/api/jobs/${{jobId}}/logs?full=1&lines=${{lines}}`);
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Failed to load full logs');
  return data.lines || [];
}}

async function refreshState() {{
  try {{
    const r = await fetch('/api/state');
    const data = await r.json();
    if (data.active_job && fullLogsEnabled()) {{
      try {{
        const lines = await fetchFullLogs(data.active_job.job_id);
        renderActive(data.active_job, lines, `full tail ${{fullLogsLines()}} lines`);
        fullLogFetchErrorShown = false;
      }} catch (logErr) {{
        renderActive(data.active_job);
        if (!fullLogFetchErrorShown) {{
          setMsg('Full log mode fallback: ' + (logErr.message || logErr), true);
          fullLogFetchErrorShown = true;
        }}
      }}
    }} else {{
      renderActive(data.active_job);
      fullLogFetchErrorShown = false;
    }}
    renderHistory(data.jobs || []);
    renderResumable(data.jobs || []);
    refreshTraceJobOptions(data.jobs || []);
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
document.getElementById('full_logs_mode').addEventListener('change', refreshState);
document.getElementById('full_logs_lines').addEventListener('change', () => {{
  if (fullLogsEnabled()) refreshState();
}});
switchAppMode('backtesting');
refreshState();
timer = setInterval(refreshState, 3000);
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


@app.get("/api/jobs/{job_id}/logs")
async def api_job_logs(job_id: str, lines: int = LOG_TAIL_LINES, full: bool = False) -> JSONResponse:
    try:
        job = JOBS.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    if full:
        n = max(1, min(int(lines), FULL_LOG_MAX_LINES))
        out = tail_file_lines(Path(job.log_path), n)
    else:
        n = max(1, min(int(lines), LOG_MEMORY_LINES))
        out = job.tail()[-n:]

    return JSONResponse({"job_id": job_id, "mode": ("full" if full else "compact"), "lines": out})


@app.get("/api/db/health")
async def api_db_health() -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        db_path = Path(sql.db_path)
        sql.safe_execute(
            sql.cursor,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        table_names = [r[0] for r in sql.cursor.fetchall()]
        table_counts: dict[str, int] = {}
        for table in table_names:
            try:
                sql.safe_execute(sql.cursor, f"SELECT COUNT(*) FROM {sqlite_ident(table)}")
                table_counts[table] = int(sql.cursor.fetchone()[0] or 0)
            except Exception:
                table_counts[table] = -1

        try:
            sql.safe_execute(sql.cursor, "PRAGMA journal_mode;")
            journal_mode = (sql.cursor.fetchone() or ["unknown"])[0]
        except Exception:
            journal_mode = "unknown"
        try:
            sql.safe_execute(sql.cursor, "PRAGMA page_count;")
            page_count = int((sql.cursor.fetchone() or [0])[0] or 0)
            sql.safe_execute(sql.cursor, "PRAGMA page_size;")
            page_size = int((sql.cursor.fetchone() or [0])[0] or 0)
        except Exception:
            page_count, page_size = 0, 0

        return JSONResponse(
            {
                "db_path": str(db_path),
                "exists": db_path.exists(),
                "size_bytes": (db_path.stat().st_size if db_path.exists() else 0),
                "modified_at": (
                    datetime.fromtimestamp(db_path.stat().st_mtime).isoformat(timespec="seconds")
                    if db_path.exists()
                    else None
                ),
                "journal_mode": journal_mode,
                "page_count": page_count,
                "page_size": page_size,
                "table_counts": table_counts,
            }
        )
    finally:
        sql.close_connection()


@app.get("/api/db/schema")
async def api_db_schema() -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        sql.safe_execute(
            sql.cursor,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        table_names = [r[0] for r in sql.cursor.fetchall()]
        tables: list[dict[str, Any]] = []
        for table in table_names:
            sql.safe_execute(sql.cursor, f"PRAGMA table_info({sqlite_ident(table)})")
            cols_raw = sql.cursor.fetchall()
            columns = [
                {
                    "cid": c[0],
                    "name": c[1],
                    "type": c[2],
                    "notnull": bool(c[3]),
                    "default": c[4],
                    "pk": bool(c[5]),
                }
                for c in cols_raw
            ]
            row_count = 0
            try:
                sql.safe_execute(sql.cursor, f"SELECT COUNT(*) FROM {sqlite_ident(table)}")
                row_count = int(sql.cursor.fetchone()[0] or 0)
            except Exception:
                row_count = -1
            tables.append({"name": table, "row_count": row_count, "columns": columns})

        return JSONResponse({"tables": tables, "default_table": (tables[0]["name"] if tables else None)})
    finally:
        sql.close_connection()


@app.get("/api/db/table")
async def api_db_table(
    table: str,
    page: int = 1,
    page_size: int = DB_DEFAULT_PAGE_SIZE,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
) -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        p, s = clamp_page_values(page, page_size)
        sql.safe_execute(
            sql.cursor,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        table_names = {r[0] for r in sql.cursor.fetchall()}
        if table not in table_names:
            raise HTTPException(status_code=404, detail=f"Unknown table: {table}")

        sql.safe_execute(sql.cursor, f"PRAGMA table_info({sqlite_ident(table)})")
        cols_raw = sql.cursor.fetchall()
        columns = [c[1] for c in cols_raw]
        if not columns:
            return JSONResponse(
                {
                    "table": table,
                    "columns": [],
                    "rows": [],
                    "page": p,
                    "page_size": s,
                    "total_rows": 0,
                    "total_pages": 0,
                }
            )

        sql.safe_execute(sql.cursor, f"SELECT COUNT(*) FROM {sqlite_ident(table)}")
        total = int(sql.cursor.fetchone()[0] or 0)
        offset = (p - 1) * s

        sort_col = sort_by if (sort_by in columns) else columns[0]
        dir_sql = "ASC" if str(sort_dir).lower() == "asc" else "DESC"
        q = (
            f"SELECT * FROM {sqlite_ident(table)} "
            f"ORDER BY {sqlite_ident(sort_col)} {dir_sql} LIMIT ? OFFSET ?"
        )
        sql.safe_execute(sql.cursor, q, (s, offset))
        rows = fetchall_dicts(sql.cursor)

        return JSONResponse(
            {
                "table": table,
                "columns": columns,
                "rows": rows,
                "page": p,
                "page_size": s,
                "total_rows": total,
                "total_pages": ((total + s - 1) // s if s else 0),
            }
        )
    finally:
        sql.close_connection()


@app.get("/api/db/backtest_runs")
async def api_db_backtest_runs(
    page: int = 1,
    page_size: int = DB_DEFAULT_PAGE_SIZE,
    pair: str = "",
    timeframe: str = "",
    strategy_id: Optional[int] = None,
    strategy_name: str = "",
    phase: str = "",
    date_from: str = "",
    date_to: str = "",
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    sort_by: str = "score",
    sort_dir: str = "desc",
) -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        p, s = clamp_page_values(page, page_size)
        from_sql = """
            FROM BacktestRuns br
            JOIN ForexPairs fp ON br.Pair_ID = fp.id
            JOIN StrategyConfigurations sc ON br.Strategy_ID = sc.id
            LEFT JOIN CompositeScores cs ON cs.id = (
                SELECT c2.id
                FROM CompositeScores c2
                WHERE c2.Backtest_ID = br.id
                ORDER BY c2.Score DESC, c2.id DESC
                LIMIT 1
            )
        """
        where: list[str] = []
        params: list[Any] = []
        if pair:
            where.append("fp.symbol = ?")
            params.append(pair)
        if timeframe:
            where.append("br.Timeframe = ?")
            params.append(timeframe)
        if strategy_id is not None:
            where.append("sc.id = ?")
            params.append(int(strategy_id))
        if strategy_name:
            where.append("sc.name LIKE ?")
            params.append(f"%{strategy_name.strip()}%")
        if phase in {"phase1", "phase2", "phase3"}:
            where.append("cs.Study_Name LIKE ?")
            params.append(f"{phase}_%")
        if date_from:
            where.append("br.Trading_Start_Date >= ?")
            params.append(date_from)
        if date_to:
            where.append("br.Trading_Start_Date <= ?")
            params.append(date_to)
        if score_min is not None:
            where.append("COALESCE(cs.Score, -1e18) >= ?")
            params.append(float(score_min))
        if score_max is not None:
            where.append("COALESCE(cs.Score, 1e18) <= ?")
            params.append(float(score_max))
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        sql.safe_execute(sql.cursor, f"SELECT COUNT(*) {from_sql} {where_sql}", tuple(params))
        total = int((sql.cursor.fetchone() or [0])[0] or 0)

        sort_map = {
            "backtest_id": "br.id",
            "pair": "fp.symbol",
            "timeframe": "br.Timeframe",
            "strategy": "sc.name",
            "score": "cs.Score",
            "net_profit": "br.Net_Profit",
            "return_pct": "br.Total_Return_Pct",
            "sharpe": "br.Sharpe_Ratio",
            "trading_start": "br.Trading_Start_Date",
        }
        order_col, order_dir = safe_sort_clause(sort_by, sort_dir, sort_map, "score")
        offset = (p - 1) * s
        q = f"""
            SELECT
                br.id AS backtest_id,
                fp.symbol AS pair,
                br.Timeframe AS timeframe,
                br.Trading_Start_Date AS trading_start_date,
                br.Data_Start_Date AS data_start_date,
                br.Data_End_Date AS data_end_date,
                br.Net_Profit AS net_profit,
                br.Total_Return_Pct AS total_return_pct,
                br.Win_Rate AS win_rate,
                br.Profit_Factor AS profit_factor,
                br.Max_Drawdown_Pct AS max_drawdown_pct,
                br.Sharpe_Ratio AS sharpe_ratio,
                sc.id AS strategy_id,
                sc.name AS strategy_name,
                cs.Score AS score,
                cs.Study_Name AS study_name,
                cs.Exploration_Space AS exploration_space
            {from_sql}
            {where_sql}
            ORDER BY {order_col} {order_dir}, br.id DESC
            LIMIT ? OFFSET ?
        """
        sql.safe_execute(sql.cursor, q, tuple(params + [s, offset]))
        rows = fetchall_dicts(sql.cursor)
        for r in rows:
            r["phase"] = phase_from_study_name(r.get("study_name"))

        return JSONResponse(
            {
                "rows": rows,
                "page": p,
                "page_size": s,
                "total_rows": total,
                "total_pages": ((total + s - 1) // s if s else 0),
            }
        )
    finally:
        sql.close_connection()


@app.get("/api/db/studies")
async def api_db_studies(
    page: int = 1,
    page_size: int = DB_DEFAULT_PAGE_SIZE,
    pair: str = "",
    timeframe: str = "",
    exploration_space: str = "",
    stop_reason: str = "",
    study_name: str = "",
    min_trials: Optional[int] = None,
    max_trials: Optional[int] = None,
    min_best_score: Optional[float] = None,
    max_best_score: Optional[float] = None,
    sort_by: str = "best_score",
    sort_dir: str = "desc",
) -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        p, s = clamp_page_values(page, page_size)
        where: list[str] = []
        params: list[Any] = []
        if pair:
            where.append("Pair = ?")
            params.append(pair)
        if timeframe:
            where.append("Timeframe = ?")
            params.append(timeframe)
        if exploration_space:
            where.append("Exploration_Space LIKE ?")
            params.append(f"%{exploration_space.strip()}%")
        if stop_reason:
            where.append("Stop_Reason LIKE ?")
            params.append(f"%{stop_reason.strip()}%")
        if study_name:
            where.append("Study_Name LIKE ?")
            params.append(f"%{study_name.strip()}%")
        if min_trials is not None:
            where.append("N_Trials >= ?")
            params.append(int(min_trials))
        if max_trials is not None:
            where.append("N_Trials <= ?")
            params.append(int(max_trials))
        if min_best_score is not None:
            where.append("Best_Score >= ?")
            params.append(float(min_best_score))
        if max_best_score is not None:
            where.append("Best_Score <= ?")
            params.append(float(max_best_score))
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        sql.safe_execute(sql.cursor, f"SELECT COUNT(*) FROM StudyMetadata {where_sql}", tuple(params))
        total = int((sql.cursor.fetchone() or [0])[0] or 0)

        sort_map = {
            "id": "id",
            "study_name": "Study_Name",
            "pair": "Pair",
            "timeframe": "Timeframe",
            "space": "Exploration_Space",
            "best_score": "Best_Score",
            "n_trials": "N_Trials",
            "n_completed": "N_Completed",
            "n_pruned": "N_Pruned",
            "total_time_sec": "Total_Time_Sec",
        }
        order_col, order_dir = safe_sort_clause(sort_by, sort_dir, sort_map, "best_score")
        offset = (p - 1) * s
        q = f"""
            SELECT
                id,
                Study_Name AS study_name,
                Pair AS pair,
                Timeframe AS timeframe,
                Exploration_Space AS exploration_space,
                Best_Score AS best_score,
                Best_Trial AS best_trial,
                N_Trials AS n_trials,
                N_Completed AS n_completed,
                N_Pruned AS n_pruned,
                Avg_Score AS avg_score,
                Std_Score AS std_score,
                Stop_Reason AS stop_reason,
                Time_to_Best AS time_to_best,
                Total_Time_Sec AS total_time_sec
            FROM StudyMetadata
            {where_sql}
            ORDER BY {order_col} {order_dir}, id DESC
            LIMIT ? OFFSET ?
        """
        sql.safe_execute(sql.cursor, q, tuple(params + [s, offset]))
        rows = fetchall_dicts(sql.cursor)
        for r in rows:
            r["phase"] = phase_from_study_name(r.get("study_name"))

        return JSONResponse(
            {
                "rows": rows,
                "page": p,
                "page_size": s,
                "total_rows": total,
                "total_pages": ((total + s - 1) // s if s else 0),
            }
        )
    finally:
        sql.close_connection()


@app.get("/api/db/composite_scores")
async def api_db_composite_scores(
    page: int = 1,
    page_size: int = DB_DEFAULT_PAGE_SIZE,
    pair: str = "",
    timeframe: str = "",
    study_name: str = "",
    exploration_space: str = "",
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    sort_by: str = "score",
    sort_dir: str = "desc",
) -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        p, s = clamp_page_values(page, page_size)
        from_sql = """
            FROM CompositeScores cs
            LEFT JOIN BacktestRuns br ON br.id = cs.Backtest_ID
            LEFT JOIN ForexPairs fp ON fp.id = br.Pair_ID
            LEFT JOIN StrategyConfigurations sc ON sc.id = br.Strategy_ID
        """
        where: list[str] = []
        params: list[Any] = []
        if pair:
            where.append("fp.symbol = ?")
            params.append(pair)
        if timeframe:
            where.append("br.Timeframe = ?")
            params.append(timeframe)
        if study_name:
            where.append("cs.Study_Name LIKE ?")
            params.append(f"%{study_name.strip()}%")
        if exploration_space:
            where.append("cs.Exploration_Space LIKE ?")
            params.append(f"%{exploration_space.strip()}%")
        if score_min is not None:
            where.append("cs.Score >= ?")
            params.append(float(score_min))
        if score_max is not None:
            where.append("cs.Score <= ?")
            params.append(float(score_max))
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        sql.safe_execute(sql.cursor, f"SELECT COUNT(*) {from_sql} {where_sql}", tuple(params))
        total = int((sql.cursor.fetchone() or [0])[0] or 0)

        sort_map = {
            "id": "cs.id",
            "score": "cs.Score",
            "trial_id": "cs.Trial_ID",
            "study_name": "cs.Study_Name",
            "timestamp": "cs.Timestamp",
            "pair": "fp.symbol",
            "timeframe": "br.Timeframe",
            "net_profit": "br.Net_Profit",
            "return_pct": "br.Total_Return_Pct",
        }
        order_col, order_dir = safe_sort_clause(sort_by, sort_dir, sort_map, "score")
        offset = (p - 1) * s
        q = f"""
            SELECT
                cs.id AS composite_id,
                cs.Score AS score,
                cs.Trial_ID AS trial_id,
                cs.Study_Name AS study_name,
                cs.Exploration_Space AS exploration_space,
                cs.Timestamp AS score_timestamp,
                cs.Backtest_ID AS backtest_id,
                fp.symbol AS pair,
                br.Timeframe AS timeframe,
                sc.id AS strategy_id,
                sc.name AS strategy_name,
                br.Net_Profit AS net_profit,
                br.Total_Return_Pct AS total_return_pct
            {from_sql}
            {where_sql}
            ORDER BY {order_col} {order_dir}, cs.id DESC
            LIMIT ? OFFSET ?
        """
        sql.safe_execute(sql.cursor, q, tuple(params + [s, offset]))
        rows = fetchall_dicts(sql.cursor)
        for r in rows:
            r["phase"] = phase_from_study_name(r.get("study_name"))

        return JSONResponse(
            {
                "rows": rows,
                "page": p,
                "page_size": s,
                "total_rows": total,
                "total_pages": ((total + s - 1) // s if s else 0),
            }
        )
    finally:
        sql.close_connection()


@app.get("/api/db/trades")
async def api_db_trades(
    backtest_id: int,
    page: int = 1,
    page_size: int = DB_DEFAULT_PAGE_SIZE,
    sort_by: str = "timestamp",
    sort_dir: str = "desc",
) -> JSONResponse:
    if int(backtest_id) <= 0:
        raise HTTPException(status_code=400, detail="backtest_id must be > 0")

    sql = get_backtest_sql(read_only=True)
    try:
        p, s = clamp_page_values(page, page_size)
        sql.safe_execute(
            sql.cursor,
            """
            SELECT
                br.id AS backtest_id,
                fp.symbol AS pair,
                br.Timeframe AS timeframe,
                sc.id AS strategy_id,
                sc.name AS strategy_name,
                br.Trading_Start_Date AS trading_start_date,
                br.Net_Profit AS net_profit,
                br.Total_Return_Pct AS total_return_pct,
                br.Win_Rate AS win_rate,
                br.Profit_Factor AS profit_factor,
                br.Max_Drawdown_Pct AS max_drawdown_pct,
                br.Sharpe_Ratio AS sharpe_ratio
            FROM BacktestRuns br
            JOIN ForexPairs fp ON fp.id = br.Pair_ID
            JOIN StrategyConfigurations sc ON sc.id = br.Strategy_ID
            WHERE br.id = ?
            """,
            (int(backtest_id),),
        )
        run = fetchone_dict(sql.cursor)
        if not run:
            raise HTTPException(status_code=404, detail=f"Unknown backtest_id: {backtest_id}")

        sql.safe_execute(sql.cursor, "SELECT COUNT(*) FROM Trades WHERE Backtest_ID = ?", (int(backtest_id),))
        total = int((sql.cursor.fetchone() or [0])[0] or 0)

        sort_map = {
            "id": "id",
            "timestamp": "Timestamp",
            "exit_timestamp": "Exit_Timestamp",
            "pnl": "PnL",
            "return_pct": "Return_Pct",
            "duration": "Duration_Minutes",
            "net_pips": "Net_Pips",
            "entry": "Entry_Price",
            "exit": "Exit_Price",
        }
        order_col, order_dir = safe_sort_clause(sort_by, sort_dir, sort_map, "timestamp")
        offset = (p - 1) * s
        q = f"""
            SELECT
                id,
                Timestamp AS timestamp,
                Exit_Timestamp AS exit_timestamp,
                Direction AS direction,
                Entry_Price AS entry_price,
                Exit_Price AS exit_price,
                Units AS units,
                PnL AS pnl,
                Net_Pips AS net_pips,
                Return_Pct AS return_pct,
                Duration_Minutes AS duration_minutes,
                Commission AS commission,
                Starting_Balance AS starting_balance,
                End_Balance AS end_balance
            FROM Trades
            WHERE Backtest_ID = ?
            ORDER BY {order_col} {order_dir}, id DESC
            LIMIT ? OFFSET ?
        """
        sql.safe_execute(sql.cursor, q, (int(backtest_id), s, offset))
        rows = fetchall_dicts(sql.cursor)

        return JSONResponse(
            {
                "run": run,
                "rows": rows,
                "page": p,
                "page_size": s,
                "total_rows": total,
                "total_pages": ((total + s - 1) // s if s else 0),
            }
        )
    finally:
        sql.close_connection()


@app.get("/api/db/strategy/{strategy_id}")
async def api_db_strategy(strategy_id: int, limit_runs: int = 100) -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        lim = max(1, min(int(limit_runs), 500))
        sql.safe_execute(
            sql.cursor,
            "SELECT id, name, description, parameters FROM StrategyConfigurations WHERE id = ?",
            (int(strategy_id),),
        )
        row = sql.cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Unknown strategy_id: {strategy_id}")
        strategy = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "parameters_raw": row[3],
            "parameters": parse_json_maybe(row[3], {}),
        }
        sql.safe_execute(
            sql.cursor,
            f"""
            SELECT
                br.id AS backtest_id,
                fp.symbol AS pair,
                br.Timeframe AS timeframe,
                br.Trading_Start_Date AS trading_start_date,
                br.Net_Profit AS net_profit,
                br.Total_Return_Pct AS total_return_pct,
                (
                    SELECT c.Score
                    FROM CompositeScores c
                    WHERE c.Backtest_ID = br.id
                    ORDER BY c.Score DESC, c.id DESC
                    LIMIT 1
                ) AS score
            FROM BacktestRuns br
            JOIN ForexPairs fp ON fp.id = br.Pair_ID
            WHERE br.Strategy_ID = ?
            ORDER BY score IS NULL ASC, score DESC, br.id DESC
            LIMIT ?
            """,
            (int(strategy_id), lim),
        )
        linked_runs = fetchall_dicts(sql.cursor)
        return JSONResponse({"strategy": strategy, "linked_runs": linked_runs})
    finally:
        sql.close_connection()


@app.get("/api/db/top_strategies")
async def api_db_top_strategies(
    mode: str = "percent",
    exploration_space: str = "default",
    top_percent: float = 10.0,
    top_n: int = 25,
    exclude_spaces: str = "generalization_test",
) -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        rows: list[dict[str, Any]] = []
        if mode == "percent":
            rows = sql.select_top_percent_strategies(
                exploration_space=exploration_space,
                top_percent=float(top_percent),
            )
        elif mode == "top_n":
            rows = sql.select_top_n_strategies_across_studies(
                exclude_exploration_spaces=split_csv(exclude_spaces),
                top_n=max(1, int(top_n)),
            )
        elif mode == "best_pair":
            rows = sql.get_best_strategy_per_pair_with_metrics()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")
        return JSONResponse({"mode": mode, "rows": rows})
    finally:
        sql.close_connection()


@app.get("/api/db/run_config/{backtest_id}")
async def api_db_run_config(backtest_id: int) -> JSONResponse:
    sql = get_backtest_sql(read_only=True)
    try:
        sql.safe_execute(
            sql.cursor,
            """
            SELECT
                br.id AS backtest_id,
                fp.symbol AS pair,
                br.Timeframe AS timeframe,
                sc.id AS strategy_id,
                sc.name AS strategy_name,
                (
                    SELECT c.Study_Name
                    FROM CompositeScores c
                    WHERE c.Backtest_ID = br.id
                    ORDER BY c.Score DESC, c.id DESC
                    LIMIT 1
                ) AS study_name
            FROM BacktestRuns br
            JOIN ForexPairs fp ON fp.id = br.Pair_ID
            JOIN StrategyConfigurations sc ON sc.id = br.Strategy_ID
            WHERE br.id = ?
            """,
            (int(backtest_id),),
        )
        row = fetchone_dict(sql.cursor)
        if not row:
            raise HTTPException(status_code=404, detail=f"Unknown backtest_id: {backtest_id}")

        d = UI_DEFAULTS.get("default", {})
        trials = dict(d.get("trials_by_timeframe", {}))
        tf = str(row.get("timeframe", "")).strip()
        if tf and tf not in trials:
            fallback = 200
            if trials:
                try:
                    fallback = int(next(iter(trials.values())))
                except Exception:
                    fallback = 200
            trials[tf] = max(1, fallback)

        payload = {
            "strategies": [row.get("strategy_name")],
            "pairs": [row.get("pair")],
            "phase12_timeframes": ([tf] if tf else []),
            "phase3_timeframes": [],
            "seeds": list(d.get("seeds", [42])),
            "n_processes": int(d.get("n_processes", 1)),
            "phase2_top_percent": float(d.get("phase2_top_percent", 10.0)),
            "phase3_top_n": int(d.get("phase3_top_n", 30)),
            "enable_phase1": True,
            "enable_phase2": False,
            "enable_phase3": False,
            "trials_by_timeframe": trials,
            "trials_by_timeframe_text": trials_to_text(trials),
        }
        return JSONResponse(
            {
                "source": row,
                "source_phase": phase_from_study_name(row.get("study_name")),
                "payload": payload,
                "note": "Reconstructed payload from BacktestRuns + UI defaults.",
            }
        )
    finally:
        sql.close_connection()


@app.get("/api/db/trace/{job_id}")
async def api_db_trace(job_id: str) -> JSONResponse:
    try:
        job = JOBS.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    study_names = list(job.study_names)
    if not study_names:
        return JSONResponse({"job_id": job_id, "study_names": [], "studies": [], "backtests": []})

    sql = get_backtest_sql(read_only=True)
    try:
        placeholders = ",".join("?" for _ in study_names)
        sql.safe_execute(
            sql.cursor,
            f"""
            SELECT
                id,
                Study_Name AS study_name,
                Pair AS pair,
                Timeframe AS timeframe,
                Exploration_Space AS exploration_space,
                Best_Score AS best_score,
                N_Trials AS n_trials,
                N_Completed AS n_completed,
                N_Pruned AS n_pruned,
                Stop_Reason AS stop_reason
            FROM StudyMetadata
            WHERE Study_Name IN ({placeholders})
            ORDER BY Best_Score DESC, id DESC
            """,
            tuple(study_names),
        )
        studies = fetchall_dicts(sql.cursor)

        sql.safe_execute(
            sql.cursor,
            f"""
            SELECT
                cs.Study_Name AS study_name,
                br.id AS backtest_id,
                fp.symbol AS pair,
                br.Timeframe AS timeframe,
                sc.id AS strategy_id,
                sc.name AS strategy_name,
                MAX(cs.Score) AS best_score,
                COUNT(cs.id) AS score_rows
            FROM CompositeScores cs
            JOIN BacktestRuns br ON br.id = cs.Backtest_ID
            JOIN ForexPairs fp ON fp.id = br.Pair_ID
            JOIN StrategyConfigurations sc ON sc.id = br.Strategy_ID
            WHERE cs.Study_Name IN ({placeholders})
            GROUP BY cs.Study_Name, br.id, fp.symbol, br.Timeframe, sc.id, sc.name
            ORDER BY best_score DESC, backtest_id DESC
            LIMIT 1000
            """,
            tuple(study_names),
        )
        backtests = fetchall_dicts(sql.cursor)

        return JSONResponse(
            {
                "job_id": job_id,
                "study_names": study_names,
                "studies": studies,
                "backtests": backtests,
            }
        )
    finally:
        sql.close_connection()


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
        job = JOBS.resume(job_id)
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

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("[UI WORKER] Optuna logging set to WARNING for UI mode", flush=True)
    except Exception:
        pass

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
