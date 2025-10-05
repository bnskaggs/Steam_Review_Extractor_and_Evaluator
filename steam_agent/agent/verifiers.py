"""Validation helpers for the Steam agent pipeline."""

from __future__ import annotations

from typing import Dict, Mapping, Set, Tuple


def verify_topic_density(
    blank_pct: float,
    lang_counts: Mapping[str, int] | None,
    supported_langs: Set[str] | None = None,
    target_supported_rate: float = 0.70,
    max_abs_slack: float = 0.20,
    min_rows_for_strict: int = 500,
    total_rows: int | None = None,
) -> Tuple[bool, Dict[str, float | bool]]:
    """Evaluate whether the topic density is acceptable.

    Parameters
    ----------
    blank_pct:
        Share of rows that received no topic label from classification.
    lang_counts:
        Mapping of language code to row counts observed during prepare.
    supported_langs:
        Languages supported by the taxonomy. Defaults to ``{"en"}``.
    target_supported_rate:
        Expected label rate for supported languages. Defaults to 70%.
    max_abs_slack:
        Absolute slack permitted before flagging as a failure.
    min_rows_for_strict:
        Number of rows required before applying the strict threshold.
    total_rows:
        Optional total row count; falls back to the sum of ``lang_counts``.

    Returns
    -------
    tuple
        ``(passed, info)`` where ``info`` provides context for reporting.
    """

    if lang_counts is None:
        lang_counts = {}

    supported_langs = supported_langs or {"en"}

    total = int(sum(lang_counts.values()))
    if total <= 0:
        total = 1

    supported = int(sum(lang_counts.get(lang, 0) for lang in supported_langs))
    supported_share = supported / total

    actual_labeled = max(0.0, min(1.0, 1.0 - float(blank_pct or 0.0)))
    expected_overall = supported_share * float(target_supported_rate)

    total_for_strict = total_rows if total_rows is not None else total
    strict = (total_for_strict or 0) >= int(min_rows_for_strict)

    if strict:
        threshold = expected_overall - float(max_abs_slack)
    else:
        threshold = expected_overall - (float(max_abs_slack) + 0.10)
    threshold = max(0.0, threshold)

    passed = actual_labeled >= threshold

    info: Dict[str, float | bool] = {
        "supported_share": supported_share,
        "expected_overall": expected_overall,
        "actual_labeled": actual_labeled,
        "strict": strict,
    }

    return passed, info


__all__ = ["verify_topic_density"]
