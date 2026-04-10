from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.config import (
    DEFAULT_SESSION_TEMPLATE_BY_TIMEFRAME,
    SESSION_TEMPLATE_DEFINITIONS,
    SESSION_TEMPLATE_TIMEZONE,
)


def default_execution_config_for_timeframe(timeframe: str | None) -> dict:
    tf = str(timeframe or "").strip()
    default_template = DEFAULT_SESSION_TEMPLATE_BY_TIMEFRAME.get(tf, "full_24x5")
    return {
        "session_template": default_template,
        "session_timezone": SESSION_TEMPLATE_TIMEZONE,
        "entry_session_only": True,
    }


def sanitize_execution_config(config: dict | None, timeframe: str | None) -> dict:
    out = default_execution_config_for_timeframe(timeframe)
    if not isinstance(config, dict):
        return out

    template = config.get("session_template")
    timezone = config.get("session_timezone")
    entry_only = config.get("entry_session_only")

    if isinstance(template, str) and template.strip():
        out["session_template"] = template.strip()
    if isinstance(timezone, str) and timezone.strip():
        out["session_timezone"] = timezone.strip()
    if entry_only is not None:
        out["entry_session_only"] = bool(entry_only)

    return out


def _parse_hhmm_to_minutes(value: str) -> int | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if raw == "24:00":
        return 24 * 60
    try:
        h_str, m_str = raw.split(":")
        h = int(h_str)
        m = int(m_str)
    except Exception:
        return None
    if h < 0 or h > 23 or m < 0 or m > 59:
        return None
    return h * 60 + m


def _normalize_days(days: Iterable[int] | None) -> list[int]:
    if days is None:
        return []
    out = []
    seen = set()
    for d in days:
        try:
            di = int(d)
        except Exception:
            continue
        if di < 0 or di > 6:
            continue
        if di not in seen:
            out.append(di)
            seen.add(di)
    return out


def _expand_window(window: dict) -> list[dict]:
    days = _normalize_days(window.get("days"))
    if not days:
        return []

    start_min = _parse_hhmm_to_minutes(window.get("start"))
    end_min = _parse_hhmm_to_minutes(window.get("end"))
    if start_min is None or end_min is None:
        return []

    # Zero-width interval means "off".
    if start_min == end_min:
        return []

    if start_min < end_min:
        return [{"days": days, "start_min": start_min, "end_min": end_min}]

    # Cross-midnight interval: split into [start, 24:00) and [00:00, end) on next day.
    next_days = sorted({(d + 1) % 7 for d in days})
    out = []
    if start_min < 24 * 60:
        out.append({"days": days, "start_min": start_min, "end_min": 24 * 60})
    if end_min > 0:
        out.append({"days": next_days, "start_min": 0, "end_min": end_min})
    return out


def _compiled_template_windows(template_name: str) -> list[dict]:
    windows = SESSION_TEMPLATE_DEFINITIONS.get(template_name)
    if not isinstance(windows, list):
        return []
    out: list[dict] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        out.extend(_expand_window(window))
    return out


def build_entry_allowed_mask(
    timestamps: pd.Series | np.ndarray | list,
    execution_config: dict | None,
    *,
    timeframe: str | None = None,
) -> np.ndarray | None:
    cfg = sanitize_execution_config(execution_config, timeframe)
    template = str(cfg.get("session_template") or "").strip()
    if not template or template.lower() in {"none", "off", "disabled", "all"}:
        return None

    windows = _compiled_template_windows(template)
    if not windows:
        logging.warning(
            "Unknown/empty session template '%s'. Trading-hours gating disabled.",
            template,
        )
        return None

    timezone = str(cfg.get("session_timezone") or SESSION_TEMPLATE_TIMEZONE).strip()

    ts_utc = pd.to_datetime(pd.Series(timestamps), errors="coerce", utc=True)
    if ts_utc.empty:
        return np.array([], dtype=bool)

    try:
        ts_local = ts_utc.dt.tz_convert(timezone)
    except Exception as e:
        logging.warning(
            "Invalid session timezone '%s' (%s). Trading-hours gating disabled.",
            timezone,
            e,
        )
        return None

    valid = ts_local.notna().to_numpy(dtype=bool)
    weekday = ts_local.dt.weekday.to_numpy(dtype=np.int8)
    minute = (ts_local.dt.hour * 60 + ts_local.dt.minute).to_numpy(dtype=np.int16)

    mask = np.zeros(len(ts_local), dtype=bool)
    for window in windows:
        days = np.array(window["days"], dtype=np.int8)
        start_min = int(window["start_min"])
        end_min = int(window["end_min"])
        if start_min >= end_min:
            continue
        in_day = np.isin(weekday, days)
        in_time = (minute >= start_min) & (minute < end_min)
        mask |= valid & in_day & in_time

    return mask
