"""Reporting helpers for the Steam agent pipeline."""

from __future__ import annotations

from typing import Mapping


def _format_language_mix(lang_counts: Mapping[str, int]) -> str:
    total = sum(int(count) for count in lang_counts.values())
    if total <= 0:
        return "Language mix: (none)"

    parts: list[str] = []
    for lang, raw_count in sorted(lang_counts.items(), key=lambda kv: (-int(kv[1]), kv[0])):
        count = int(raw_count)
        share = count / total
        parts.append(f"{lang}: {share:.1%} ({count})")
    return "Language mix: " + ", ".join(parts)


def render_topic_density_footer(
    lang_counts: Mapping[str, int] | None,
    density_info: Mapping[str, float | bool] | None,
) -> str:
    """Return a footer block describing language mix and coverage expectations."""

    lang_counts = lang_counts or {}
    density_info = density_info or {}

    lines: list[str] = []
    lines.append(_format_language_mix(lang_counts))

    supported_share = float(density_info.get("supported_share", 0.0) or 0.0)
    lines.append(f"Supported-language share: {supported_share:.1%}")

    actual = float(density_info.get("actual_labeled", 0.0) or 0.0)
    expected = float(density_info.get("expected_overall", 0.0) or 0.0)
    lines.append(
        "Coverage: {actual:.1%} overall (expected ~{expected:.1%} based on {supported:.1%} supported-language share).".format(
            actual=actual,
            expected=expected,
            supported=supported_share,
        )
    )

    return "\n".join(lines)


__all__ = ["render_topic_density_footer"]
