"""Lightweight execution tracer using sys.settrace.

Records per-function entry/exit events and hashes them into a compact
signature suitable for equivalence-class comparison between two patches.

Usage::

    tracer = ExecutionTracer()
    with tracer.trace():
        exec(test_code, namespace)
    trace = tracer.collect()
    print(trace.path_hash)
"""

from __future__ import annotations

import hashlib
import sys
import textwrap
import traceback
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class FunctionEvent:
    """A single entry or exit event captured during tracing."""

    kind: str          # "call" | "return" | "exception"
    filename: str
    funcname: str
    lineno: int
    return_repr: str = ""   # repr of return value (truncated)


@dataclass
class ExecutionTrace:
    """Aggregated trace for one test-suite execution."""

    events: list[FunctionEvent] = field(default_factory=list)
    covered_lines: dict[str, set[int]] = field(default_factory=dict)  # file → lines
    error: str | None = None

    @property
    def path_hash(self) -> str:
        """SHA-256 of the call-sequence fingerprint (filename+func+kind)."""
        fingerprint = "|".join(
            f"{e.filename}:{e.funcname}:{e.kind}" for e in self.events
        )
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    @property
    def covered_files(self) -> set[str]:
        return set(self.covered_lines.keys())

    @property
    def covered_functions(self) -> set[tuple[str, str]]:
        """Set of (filename, funcname) pairs exercised during the trace."""
        return {(e.filename, e.funcname) for e in self.events if e.kind == "call"}

    def diff(self, other: "ExecutionTrace") -> "TraceDiff":
        """Compute symmetric diff between two traces."""
        self_files = set(self.covered_lines.keys())
        other_files = set(other.covered_lines.keys())
        self_funcs = self.covered_functions
        other_funcs = other.covered_functions
        return TraceDiff(
            new_files=other_files - self_files,
            removed_files=self_files - other_files,
            new_functions=other_funcs - self_funcs,
            path_changed=self.path_hash != other.path_hash,
        )


@dataclass
class TraceDiff:
    """Summary of differences between two execution traces."""

    new_files: set[str]                    # files newly covered in `other`
    removed_files: set[str]               # files no longer covered in `other`
    new_functions: set[tuple[str, str]]   # (file, func) pairs new in `other`
    path_changed: bool                    # call-sequence hash differs

    @property
    def has_scope_creep(self) -> bool:
        """New files OR new functions exercised → potential boundary violation."""
        return bool(self.new_files) or bool(self.new_functions)


class ExecutionTracer:
    """Collects an :class:`ExecutionTrace` for arbitrary Python code.

    Parameters
    ----------
    filter_prefix:
        Only record events from files whose path contains this prefix.
        Useful to exclude stdlib/site-packages noise.
    max_events:
        Hard cap on recorded events to avoid unbounded memory use.
    """

    def __init__(
        self,
        filter_prefix: str = "",
        max_events: int = 10_000,
    ) -> None:
        self._filter_prefix = filter_prefix
        self._max_events = max_events
        self._events: list[FunctionEvent] = []
        self._covered: dict[str, set[int]] = {}
        self._active = False

    # ------------------------------------------------------------------
    # sys.settrace callback
    # ------------------------------------------------------------------

    def _trace_calls(
        self,
        frame: types.FrameType,
        event: str,
        arg: Any,
    ) -> Any:
        if len(self._events) >= self._max_events:
            return None

        filename: str = frame.f_code.co_filename
        if self._filter_prefix and self._filter_prefix not in filename:
            return self._trace_calls  # keep tracing but don't record

        funcname: str = frame.f_code.co_name
        lineno: int = frame.f_lineno

        # Line coverage
        if filename not in self._covered:
            self._covered[filename] = set()
        self._covered[filename].add(lineno)

        if event == "call":
            self._events.append(
                FunctionEvent(
                    kind="call",
                    filename=filename,
                    funcname=funcname,
                    lineno=lineno,
                )
            )
            return self._trace_calls

        if event == "return":
            self._events.append(
                FunctionEvent(
                    kind="return",
                    filename=filename,
                    funcname=funcname,
                    lineno=lineno,
                    return_repr=repr(arg)[:120],
                )
            )

        if event == "exception":
            exc_type, exc_val, _ = arg
            self._events.append(
                FunctionEvent(
                    kind="exception",
                    filename=filename,
                    funcname=funcname,
                    lineno=lineno,
                    return_repr=f"{exc_type.__name__}: {exc_val}",
                )
            )

        return self._trace_calls

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @contextmanager
    def trace(self) -> Generator[None, None, None]:
        """Context manager that activates tracing for the enclosed block."""
        self._events.clear()
        self._covered.clear()
        self._active = True
        old_trace = sys.gettrace()
        sys.settrace(self._trace_calls)
        try:
            yield
        finally:
            sys.settrace(old_trace)
            self._active = False

    def collect(self) -> ExecutionTrace:
        """Return the :class:`ExecutionTrace` from the last ``trace()`` block."""
        return ExecutionTrace(
            events=list(self._events),
            covered_lines={k: set(v) for k, v in self._covered.items()},
        )


# ---------------------------------------------------------------------------
# Convenience: run test source and capture trace
# ---------------------------------------------------------------------------

def trace_test_execution(
    patch_files: dict[str, str],
    test_source: str,
    filter_prefix: str = "",
) -> ExecutionTrace:
    """Execute ``test_source`` against ``patch_files`` and return a trace.

    ``patch_files`` is written into an in-memory namespace via ``exec``; this
    is intentionally sandboxed and suitable only for *trusted* code.

    Parameters
    ----------
    patch_files:
        Dict mapping module-like names to their source strings.
    test_source:
        Source of the test file to execute.
    filter_prefix:
        Passed to :class:`ExecutionTracer`.
    """
    namespace: dict[str, Any] = {}

    # Compile and exec each patched file into the namespace first.
    for _fname, src in patch_files.items():
        try:
            exec(compile(src, _fname, "exec"), namespace)  # noqa: S102
        except Exception:
            return ExecutionTrace(error=traceback.format_exc())

    tracer = ExecutionTracer(filter_prefix=filter_prefix)
    error: str | None = None

    with tracer.trace():
        try:
            exec(  # noqa: S102
                compile(textwrap.dedent(test_source), "<test>", "exec"),
                namespace,
            )
        except Exception:
            error = traceback.format_exc()

    trace = tracer.collect()
    trace.error = error
    return trace
