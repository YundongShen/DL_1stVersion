"""Core data types for training and evaluation pairs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingPair:
    """One contrastive training or evaluation example.

    Fields
    ------
    instance_id:
        Source SWE-bench instance identifier, for traceability.
    issue_text:
        Natural-language problem description (anchor context).
    test_suite_signature:
        Sorted, newline-joined test function names (anchor context).
    old_code:
        Concatenated source files before any patch is applied.
    positive_diff:
        In-boundary diff (correct fix, does not exceed test-induced scope).
    negative_diff:
        Out-of-boundary diff (fixes the bug but exhibits scope creep).
    positive_source:
        ``"gold"`` for human SWE-bench patches;
        ``"controlled_llm"`` for LLM-generated patches verified in-boundary.
    negative_source:
        Always ``"llm_unconstrained"`` — LLM output filtered for scope creep.
    """

    instance_id: str
    issue_text: str
    test_suite_signature: str
    old_code: str
    positive_diff: str
    negative_diff: str
    positive_source: str   # "gold" | "controlled_llm"
    negative_source: str   # "llm_unconstrained"
