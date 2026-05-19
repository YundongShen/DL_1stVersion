"""Filtering logic for scope creep detection and boundary verification."""

from __future__ import annotations

import re

from .execution_tracer import ExecutionTrace
from .utils import line_count_ratio


def _diff_files(diff: str) -> set[str]:
    return {
        m.group(1).strip()
        for m in re.finditer(r"^--- (?:a/)?(.*)", diff, re.MULTILINE)
        if m.group(1).strip() != "/dev/null"
    }


def _diff_added_lines(diff: str) -> int:
    return sum(
        1 for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def has_scope_creep(
    reference_trace: ExecutionTrace,
    candidate_trace: ExecutionTrace,
    *,
    gold_diff: str | None = None,
    unctr_diff: str | None = None,
) -> bool:
    """Return True if the candidate patch has scope creep relative to reference.

    When diffs are provided, uses diff-based comparison (robust).
    Falls back to execution-trace comparison if diffs are not supplied.

    Diff-based criteria:
    - unconstrained touches a file the gold patch did not, OR
    - unconstrained adds ≥5 more lines than gold
    """
    if gold_diff is not None and unctr_diff is not None:
        unctr_files = _diff_files(unctr_diff)
        gold_files = _diff_files(gold_diff)
        if unctr_files - gold_files:
            return True
        return _diff_added_lines(unctr_diff) >= _diff_added_lines(gold_diff) + 5
    # Legacy fallback: execution-trace based
    diff = reference_trace.diff(candidate_trace)
    return diff.has_scope_creep


def is_trace_subset(candidate_trace: ExecutionTrace, gold_trace: ExecutionTrace) -> bool:
    """Return True if candidate's covered files are a subset of gold's covered files.

    Used to verify that a controlled LLM patch stays within the gold patch's scope,
    qualifying it as a clean positive for the zero-shot validation set.

    Parameters
    ----------
    candidate_trace:
        Trace of the controlled LLM patch run.
    gold_trace:
        Trace of the human gold patch run (the ground-truth boundary).
    """
    if gold_trace.error or candidate_trace.error:
        return False
    return candidate_trace.covered_files <= gold_trace.covered_files


def line_count_ok(diff_a: str, diff_b: str, tolerance: float = 0.5) -> bool:
    """Return True if the two diffs have similar line counts.

    Prevents trivially short negatives that would make classification too easy.
    Delegates to :func:`~data.utils.line_count_ratio`.
    """
    return line_count_ratio(diff_a, diff_b, tolerance)
