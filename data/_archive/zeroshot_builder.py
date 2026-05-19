"""Build zero-shot validation pairs: verified controlled LLM patch (positive) vs unconstrained (negative).

Design rationale
----------------
This set is held out entirely from training and used to falsify the hypothesis that the
model learned "human style vs LLM style" rather than boundary compliance.

* Positive = LLM controlled patch, verified to stay within gold patch's trace scope.
  Crucially, this is LLM-generated — same source as the negative.

* Negative = LLM unconstrained patch, same filtering as training set.

If the model (trained only on gold positives) correctly classifies controlled LLM patches
as in-boundary despite never seeing LLM positives during training, the "style confound"
attack is empirically falsified.
"""

from __future__ import annotations

import logging

from .data_loader import DataSample
from .pair_types import TrainingPair
from .patch_generator import PatchGenerator
from .sandbox import SandboxRunner
from .trace_filter import has_scope_creep, is_trace_subset, line_count_ok
from .utils import build_test_signature, flatten_codebase

log = logging.getLogger(__name__)


class ZeroshotSetBuilder:
    """Produce zero-shot validation :class:`~data.pair_types.TrainingPair` objects.

    Parameters
    ----------
    generator:
        :class:`~data.patch_generator.PatchGenerator` for LLM calls.
    sandbox:
        :class:`~data.sandbox.SandboxRunner` for test execution.
    line_count_tolerance:
        Maximum allowed fractional difference in diff line counts.
    """

    def __init__(
        self,
        generator: PatchGenerator,
        sandbox: SandboxRunner,
        line_count_tolerance: float = 0.5,
    ) -> None:
        self._gen = generator
        self._sandbox = sandbox
        self._tol = line_count_tolerance

    def _run(self, sample: DataSample, patch_str: str | None):
        repo_path = sample.metadata.get("repo_path")
        test_ids = sample.metadata.get("fail_to_pass", [])
        prerequisite = sample.metadata.get("test_patch", "")
        if not repo_path or not test_ids:
            return False, None
        return self._sandbox.run_with_trace(
            repo_path=repo_path,
            test_ids=test_ids,
            patch_str=patch_str,
            prerequisite_patch=prerequisite,
        )

    def build(self, sample: DataSample) -> TrainingPair | None:
        """Return a zero-shot validation :class:`TrainingPair` or ``None``.

        Steps
        -----
        1. Run gold patch to get ground-truth trace boundary.
        2. Generate and run controlled patch.  Must pass tests AND have trace
           that is a subset of gold trace (verified in-boundary).
        3. Generate and run unconstrained patch.  Must pass tests AND show
           scope creep vs controlled patch.
        4. Emit pair with controlled LLM patch as positive.
        """
        iid = sample.sample_id

        # Step 1: gold trace as boundary reference.
        gold_passed, gold_trace = self._run(sample, sample.golden_diff)
        if not gold_passed or gold_trace is None or gold_trace.error:
            log.debug("%s: gold patch failed — skipping", iid)
            return None

        # Step 2: controlled patch — must be verified in-boundary vs gold.
        ctrl_diff = self._gen.generate_controlled(sample)
        if not ctrl_diff:
            return None
        ctrl_passed, ctrl_trace = self._run(sample, ctrl_diff)
        if not ctrl_passed or ctrl_trace is None or ctrl_trace.error:
            log.debug("%s: controlled patch failed tests", iid)
            return None
        if not is_trace_subset(ctrl_trace, gold_trace):
            log.debug("%s: controlled patch exceeds gold trace — not a clean positive", iid)
            return None

        # Step 3: unconstrained patch — must show scope creep vs controlled.
        unctr_diff = self._gen.generate_unconstrained(sample)
        if not unctr_diff:
            return None
        unctr_passed, unctr_trace = self._run(sample, unctr_diff)
        if not unctr_passed or unctr_trace is None or unctr_trace.error:
            log.debug("%s: unconstrained patch failed tests", iid)
            return None
        if not has_scope_creep(ctrl_trace, unctr_trace):
            log.debug("%s: no scope creep — discarding", iid)
            return None
        if not line_count_ok(ctrl_diff, unctr_diff, self._tol):
            log.debug("%s: line-count ratio filter failed", iid)
            return None

        return TrainingPair(
            instance_id=iid,
            issue_text=sample.issue_text,
            test_suite_signature=build_test_signature(sample.test_suite),
            old_code=flatten_codebase(sample.old_codebase),
            positive_diff=ctrl_diff,
            negative_diff=unctr_diff,
            positive_source="controlled_llm",
            negative_source="llm_unconstrained",
        )
