# ml_baselines/splits.py
# -*- coding: utf-8 -*-
"""
======================== BIG PLAIN-ENGLISH OVERVIEW ========================
Time-aware splitters for ML trading that respect chronology and minimize leakage.

What's here:
1) walk_forward_splits: fixed-length rolling windows (train -> test -> slide).
2) expanding_window_splits: growing train windows with rolling fixed test.
3) blocked_kfold_time_series: blocked, time-ordered K-Fold using np.array_split.
4) single_window: build one explicit split by sizes or date.
5) Utilities: index → integer positions; monotonic index assertion.

Leakage controls:
- gap: bars skipped between train end and test start (e.g., to mirror feature shift=1).
- embargo: bars skipped AFTER each test block before the NEXT training window can end,
  i.e., we ensure the next train does not encroach into recently tested data.

Fits with:
- feature_store: features are already shifted by 1 bar to avoid look-ahead.
- labels: forward-return labels purge the last H rows; alignment happens later.
==========================================================================="""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ----------------------------- Types --------------------------------------

IndexLike = Union[pd.Index, Sequence]
Split = Tuple[np.ndarray, np.ndarray]  # (train_pos, test_pos)
SplitWithIndex = Tuple[
    np.ndarray, np.ndarray, pd.Index, pd.Index
]  # + (train_idx, test_idx)


# ----------------------------- Utilities ----------------------------------


def _assert_monotonic_index(idx: pd.Index) -> None:
    """
    Ensure the index is sorted ascending. We do NOT sort automatically to avoid
    silently desynchronizing with the caller's X/y ordering.
    """
    if not idx.is_monotonic_increasing:
        raise ValueError(
            "Index must be sorted ascending to use time-aware splits. "
            "Sort your DataFrame/Series first (e.g., df.sort_index())."
        )


def _as_positions(index_like: IndexLike) -> Tuple[pd.Index, np.ndarray]:
    """
    Return the original index (labels/timestamps) and integer positions [0..n-1].
    """
    idx = index_like if isinstance(index_like, pd.Index) else pd.Index(index_like)
    _assert_monotonic_index(idx)
    positions = np.arange(len(idx), dtype=int)
    return idx, positions


def _slice_window_positions(start: int, end: int, n: int) -> np.ndarray:
    """
    Return positions [start, end) clipped to [0, n].
    """
    start = max(0, int(start))
    end = min(n, int(end))
    if end <= start:
        return np.empty(0, dtype=int)
    return np.arange(start, end, dtype=int)


@dataclass(frozen=True)
class SplitParams:
    gap: int = 0  # bars skipped between train end and test start
    embargo: int = 0  # bars skipped AFTER test end before the next TRAIN can end


# ----------------------------- Validators ----------------------------------


def _validate_positive_int(name: str, value: Optional[int]) -> None:
    if value is None or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_have_data(n: int, train: int, test: int, gap: int) -> None:
    if train + gap + test > n:
        raise ValueError(
            f"Not enough data for one split: train({train}) + gap({gap}) + test({test}) > n({n})"
        )


# ----------------------------- Core splitters ----------------------------------


def walk_forward_splits(
    index_like: IndexLike,
    *,
    train_size: int,
    test_size: int,
    step: Optional[int] = None,
    params: SplitParams = SplitParams(),
    return_with_index: bool = False,
) -> Generator[Union[Split, SplitWithIndex], None, None]:
    """
    Fixed-size rolling windows:
      [TRAIN train_size] --gap--> [TEST test_size] --slide by step--> repeat

    Embargo semantics:
      After yielding a (train, test), the NEXT train start is advanced such that
      the next train window does not end before (end_test + embargo).

    Yields:
      (train_pos, test_pos) or (train_pos, test_pos, train_idx, test_idx)
    """
    _validate_positive_int("train_size", train_size)
    _validate_positive_int("test_size", test_size)
    if step is not None:
        _validate_positive_int("step", step)
    if params.gap < 0 or params.embargo < 0:
        raise ValueError("gap and embargo must be >= 0")

    idx, pos = _as_positions(index_like)
    n = len(idx)
    if step is None:
        step = test_size

    _validate_have_data(n, train_size, test_size, params.gap)

    start_train = 0
    while True:
        end_train = start_train + train_size
        start_test = end_train + params.gap
        end_test = start_test + test_size
        if end_test > n:
            break

        train_pos = _slice_window_positions(start_train, end_train, n)
        test_pos = _slice_window_positions(start_test, end_test, n)
        if len(train_pos) == 0 or len(test_pos) == 0:
            break

        if return_with_index:
            yield train_pos, test_pos, idx[train_pos], idx[test_pos]
        else:
            yield train_pos, test_pos

        # Next window starts by sliding 'step'
        start_train = start_train + step
        # Enforce embargo: next train must not end before (end_test + embargo)
        if params.embargo > 0:
            min_next_train_start = end_test + params.embargo - train_size
            start_train = max(start_train, min_next_train_start)


def expanding_window_splits(
    index_like: IndexLike,
    *,
    initial_train_size: int,
    test_size: int,
    step: Optional[int] = None,
    params: SplitParams = SplitParams(),
    return_with_index: bool = False,
) -> Generator[Union[Split, SplitWithIndex], None, None]:
    """
    Expanding TRAIN with fixed TEST:
      First: TRAIN [0, initial_train_size), TEST [train_end + gap, + test_size)
      Next: increase train_end by 'step' (default = test_size) and repeat.

    Embargo semantics:
      After each test, the next train_end is at least (end_test + embargo).
    """
    _validate_positive_int("initial_train_size", initial_train_size)
    _validate_positive_int("test_size", test_size)
    if step is not None:
        _validate_positive_int("step", step)
    if params.gap < 0 or params.embargo < 0:
        raise ValueError("gap and embargo must be >= 0")

    idx, pos = _as_positions(index_like)
    n = len(idx)
    if step is None:
        step = test_size

    _validate_have_data(n, initial_train_size, test_size, params.gap)

    train_end = initial_train_size
    while True:
        start_test = train_end + params.gap
        end_test = start_test + test_size
        if end_test > n:
            break

        train_pos = _slice_window_positions(0, train_end, n)
        test_pos = _slice_window_positions(start_test, end_test, n)
        if len(train_pos) == 0 or len(test_pos) == 0:
            break

        if return_with_index:
            yield train_pos, test_pos, idx[train_pos], idx[test_pos]
        else:
            yield train_pos, test_pos

        # Expand train window
        train_end = train_end + step
        if params.embargo > 0:
            train_end = max(train_end, end_test + params.embargo)


def blocked_kfold_time_series(
    index_like: IndexLike,
    *,
    n_splits: int,
    test_size: Optional[int] = None,
    params: SplitParams = SplitParams(),
    return_with_index: bool = False,
) -> Generator[Union[Split, SplitWithIndex], None, None]:
    """
    Time-ordered blocked K-Fold using np.array_split to avoid dropping the tail.

    If test_size is provided, we slice contiguous test blocks of approximately
    that size; otherwise we use np.array_split to split the full range into
    n_splits contiguous test blocks (the last blocks may differ by 1 element).

    For each fold k:
      TRAIN = all positions strictly BEFORE start_test - gap
      TEST  = the k-th contiguous block
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if params.gap < 0:
        raise ValueError("gap must be >= 0")

    idx, pos = _as_positions(index_like)
    n = len(idx)

    if test_size is not None:
        _validate_positive_int("test_size", test_size)
        # Construct test blocks of size ~test_size; include the tail as a final block
        edges = list(range(0, n, test_size))
        if edges[-1] != n:
            edges.append(n)
        blocks = [
            np.arange(edges[i], edges[i + 1], dtype=int) for i in range(len(edges) - 1)
        ]
        # If more blocks than n_splits, cap to first n_splits; else if fewer, raise.
        if len(blocks) < n_splits:
            raise ValueError(
                "With the given test_size, fewer than n_splits blocks are produced."
            )
        blocks = blocks[:n_splits]
    else:
        # Evenly split full range into n_splits contiguous blocks (diff <= 1)
        blocks = [
            np.asarray(b, dtype=int)
            for b in np.array_split(np.arange(n, dtype=int), n_splits)
        ]

    for test_pos in blocks:
        if len(test_pos) == 0:
            continue
        start_test = int(test_pos[0])
        end_train = max(0, start_test - params.gap)
        train_pos = _slice_window_positions(0, end_train, n)
        if len(train_pos) == 0:
            continue

        if return_with_index:
            yield train_pos, test_pos, idx[train_pos], idx[test_pos]
        else:
            yield train_pos, test_pos


def single_window(
    index_like: IndexLike,
    *,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    train_end_at: Optional[pd.Timestamp] = None,
    params: SplitParams = SplitParams(),
    return_with_index: bool = False,
) -> Union[Split, SplitWithIndex]:
    """
    Build one train/test split.

    Two ways:
      (A) Sizes: train_size + test_size
      (B) Date/label: train_end_at in index + test_size

    gap is respected between train end and test start.
    """
    if params.gap < 0:
        raise ValueError("gap must be >= 0")

    idx, pos = _as_positions(index_like)
    n = len(idx)

    if train_end_at is not None:
        if train_end_at not in idx:
            raise KeyError(f"train_end_at {train_end_at!r} is not in index.")
        _validate_positive_int("test_size", test_size)
        train_end_pos = int(np.searchsorted(idx, train_end_at, side="right"))
        start_test = train_end_pos + params.gap
        end_test = start_test + int(test_size)
        train_pos = _slice_window_positions(0, train_end_pos, n)
        test_pos = _slice_window_positions(start_test, end_test, n)

    else:
        _validate_positive_int("train_size", train_size)
        _validate_positive_int("test_size", test_size)
        _validate_have_data(n, int(train_size), int(test_size), params.gap)
        train_pos = _slice_window_positions(0, int(train_size), n)
        start_test = len(train_pos) + params.gap
        test_pos = _slice_window_positions(start_test, start_test + int(test_size), n)

    if return_with_index:
        return train_pos, test_pos, idx[train_pos], idx[test_pos]
    return train_pos, test_pos
