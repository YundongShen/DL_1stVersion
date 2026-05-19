"""Build training pairs: human gold patch (positive) vs LLM unconstrained (negative).

Design rationale
----------------
* Positive = SWE-bench gold patch.  Human-written, guaranteed minimal by construction
  (it is the merged PR that fixed the issue).

* Negative = LLM unconstrained patch that passes tests AND exhibits scope creep
  relative to a controlled LLM patch on the same issue.

* The controlled patch is the "ruler": it proves the LLM *can* produce a minimal fix
  for this issue, so extra changes in the unconstrained version are unjustified.

* Line-count matching ensures negatives are not trivially distinguishable by size.
"""

from __future__ import annotations

import logging

from .data_loader import DataSample
from .pair_types import TrainingPair
from .patch_generator import PatchGenerator
from .sandbox import SandboxRunner
from .trace_filter import has_scope_creep
from .utils import build_test_signature, flatten_codebase

log = logging.getLogger(__name__)


class TrainingSetBuilder:
    """Produce one :class:`~data.pair_types.TrainingPair` per DataSample.

    Parameters
    ----------
    generator:
        :class:`~data.patch_generator.PatchGenerator` for LLM calls.
    sandbox:
        :class:`~data.sandbox.SandboxRunner` for test execution.
    """

    def __init__(
        self,
        generator: PatchGenerator,
        sandbox: SandboxRunner,
    ) -> None:
        self._gen = generator
        self._sandbox = sandbox

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        sample: DataSample,
        patch_str: str | None,
    ):
        """Run tests for *patch_str* and return (passed, trace)."""
        repo_path = sample.metadata.get("repo_path")
        # Only use fail_to_pass: these have proper file::test paths and prove the patch works.
        # pass_to_pass entries are often bare function names that pytest cannot resolve.
        test_ids = sample.metadata.get("fail_to_pass", [])
        prerequisite = sample.metadata.get("test_patch", "")

        if not repo_path or not test_ids:
            log.warning("Sample %s missing repo_path or test_ids", sample.sample_id)
            return False, None

        return self._sandbox.run_with_trace(
            repo_path=repo_path,
            test_ids=test_ids,
            patch_str=patch_str,
            prerequisite_patch=prerequisite,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self, sample: DataSample) -> TrainingPair | None:
        """Return a :class:`TrainingPair` or ``None`` if filtering rejects this sample.

        Steps
        -----
        1. Run gold patch — must pass tests; its trace is the scope ruler.
        2. Generate unconstrained patch.  Must pass tests AND show scope creep
           relative to gold trace AND have similar line count to gold patch.
        3. Emit TrainingPair with gold as positive, unconstrained as negative.

        Note: the controlled patch step was removed.  The gold patch is the
        minimal fix by construction (it is the merged PR), so its execution
        trace is the most reliable boundary reference.
        """
        iid = sample.sample_id

        # Step 1: verify gold patch passes; capture its trace as the ruler.
        gold_diff = sample.golden_diff
        gold_passed, gold_trace = self._run(sample, gold_diff)
        if gold_trace is not None and gold_trace.error:
            log.info("%s: SKIP — gold trace error: %s", iid, gold_trace.error[:200])
            return None
        if not gold_passed:
            log.info("%s: SKIP — gold patch failed tests (trace_events=%d)",
                     iid, len(gold_trace.events) if gold_trace else -1)
            return None
        log.info("%s: gold OK (events=%d)", iid, len(gold_trace.events))

        # Step 2: generate and validate unconstrained patch.
        unctr_diff = self._gen.generate_unconstrained(sample)
        if not unctr_diff:
            log.info("%s: SKIP — unconstrained generation failed", iid)
            return None
        log.info("%s: unconstrained diff generated (%d lines)", iid, unctr_diff.count("\n"))
        unctr_passed, unctr_trace = self._run(sample, unctr_diff)
        if not unctr_passed or unctr_trace is None or unctr_trace.error:
            err = unctr_trace.error if unctr_trace else "no trace"
            log.info("%s: SKIP — unconstrained patch failed tests (passed=%s err=%s)", iid, unctr_passed, err)
            return None
        log.info("%s: unconstrained OK", iid)

        # Filter: unconstrained must show scope creep vs gold (diff-based).
        if not has_scope_creep(gold_trace, unctr_trace,
                               gold_diff=gold_diff, unctr_diff=unctr_diff):
            from .trace_filter import _diff_added_lines
            log.info("%s: SKIP — no scope creep (gold_added=%d unctr_added=%d)",
                     iid, _diff_added_lines(gold_diff), _diff_added_lines(unctr_diff))
            return None

        return TrainingPair(
            instance_id=iid,
            issue_text=sample.issue_text,
            test_suite_signature=build_test_signature(sample.test_suite),
            old_code=flatten_codebase(sample.old_codebase),
            positive_diff=gold_diff,
            negative_diff=unctr_diff,
            positive_source="gold",
            negative_source="llm_unconstrained",
        )
