"""Pipeline orchestration utilities for the Steam agent."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Dict, Tuple

from ..pipeline import report as report_module
from . import verifiers

__all__ = ["run_pipeline"]


def _ensure_mapping(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    if value is None:
        return {}
    return dict(value)


def _normalize_lang_counts(raw: Mapping[str, Any] | None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if raw:
        for key, value in raw.items():
            try:
                counts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    return counts


def _call_with_metrics(
    fn: Callable[..., Mapping[str, Any] | Tuple[Any, Mapping[str, Any]]],
    kwargs: Mapping[str, Any] | None = None,
) -> Tuple[Any | None, Dict[str, Any]]:
    call_kwargs = dict(kwargs or {})
    result = fn(**call_kwargs)

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        payload, metrics = result
    elif isinstance(result, Mapping):
        payload, metrics = None, result
    else:
        raise TypeError("Callable is expected to return a mapping of metrics or (payload, metrics).")

    return payload, _ensure_mapping(metrics)


def run_pipeline(
    prepare_fn: Callable[..., Mapping[str, Any] | Tuple[Any, Mapping[str, Any]]],
    classify_fn: Callable[..., Mapping[str, Any] | Tuple[Any, Mapping[str, Any]]],
    *,
    min_conf: float,
    logger: logging.Logger | None = None,
    supported_langs: set[str] | None = None,
    target_supported_rate: float = 0.70,
    max_abs_slack: float = 0.20,
    min_rows_for_strict: int = 500,
    threshold_floor: float = 0.35,
    threshold_step: float = 0.05,
    prepare_kwargs: Mapping[str, Any] | None = None,
    classify_kwargs: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the prepare + classify steps with dynamic topic-density checks."""

    logger = logger or logging.getLogger(__name__)
    supported_langs = supported_langs or {"en"}

    prepare_payload, prepare_metrics = _call_with_metrics(prepare_fn, prepare_kwargs)

    lang_counts = _normalize_lang_counts(prepare_metrics.get("lang_counts"))
    total_rows = prepare_metrics.get("rows_in")

    classify_kwargs = dict(classify_kwargs or {})
    classify_kwargs["min_conf"] = min_conf

    classify_payload, classify_metrics = _call_with_metrics(classify_fn, classify_kwargs)

    blank_pct = classify_metrics.get("blank_pct")
    if blank_pct is None:
        raise KeyError("Classification metrics must include 'blank_pct'.")

    ok, density_info = verifiers.verify_topic_density(
        blank_pct=float(blank_pct),
        lang_counts=lang_counts,
        supported_langs=supported_langs,
        target_supported_rate=target_supported_rate,
        max_abs_slack=max_abs_slack,
        min_rows_for_strict=min_rows_for_strict,
        total_rows=total_rows,
    )

    current_min_conf = min_conf

    if not ok:
        logger.info(
            "Topic coverage %.1f%% below expectation (~%.1f%%). Attempting to lower min_conf.",
            (density_info["actual_labeled"] * 100.0),
            (density_info["expected_overall"] * 100.0),
        )
        new_min_conf = max(current_min_conf - threshold_step, threshold_floor)
        if new_min_conf < current_min_conf:
            classify_kwargs["min_conf"] = new_min_conf
            classify_payload, classify_metrics = _call_with_metrics(classify_fn, classify_kwargs)
            blank_pct = classify_metrics.get("blank_pct")
            if blank_pct is None:
                raise KeyError("Classification metrics must include 'blank_pct' after retry.")

            ok, density_info = verifiers.verify_topic_density(
                blank_pct=float(blank_pct),
                lang_counts=lang_counts,
                supported_langs=supported_langs,
                target_supported_rate=target_supported_rate,
                max_abs_slack=max_abs_slack,
                min_rows_for_strict=min_rows_for_strict,
                total_rows=total_rows,
            )
            current_min_conf = new_min_conf
        else:
            logger.debug("min_conf already at floor %.2f; skipping retry.", threshold_floor)

    if not ok:
        supported_share = density_info.get("supported_share", 0.0) or 0.0
        strict = bool(density_info.get("strict", False))
        if supported_share < 0.15 or not strict:
            logger.warning(
                "Topic coverage %.1f%% remains below expectation (~%.1f%%) but continuing due to "
                "supported-language share %.1f%% and strict=%s.",
                density_info["actual_labeled"] * 100.0,
                density_info["expected_overall"] * 100.0,
                supported_share * 100.0,
                strict,
            )
        else:
            raise RuntimeError(
                "Topic density verification failed despite sufficient supported-language share."
            )

    footer = report_module.render_topic_density_footer(lang_counts, density_info)

    outputs = {
        "prepare": prepare_payload,
        "classify": classify_payload,
    }
    metrics = {
        "prepare": prepare_metrics,
        "classify": classify_metrics,
        "topic_density": density_info,
        "min_conf": current_min_conf,
        "footer": footer,
    }

    return outputs, metrics
