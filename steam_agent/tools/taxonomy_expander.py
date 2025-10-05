"""Utilities for expanding taxonomy keywords from review data."""
from __future__ import annotations

import argparse
import collections
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import yaml
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

WINDOW_RADIUS = 6
NGRAM_RANGE = (1, 3)
STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "game",
    "games",
    "play",
    "player",
    "players",
    "steam",
    "really",
    "review",
    "reviews",
    "thing",
    "things",
}

_TOKEN_REGEX = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass
class TopicConfig:
    name: str
    seeds: set[str]
    existing_terms: set[str]


@dataclass
class TopicSuggestions:
    add: list[str]
    by_language: dict[str, list[str]] | None = None


class TaxonomyExpanderError(Exception):
    """Custom error raised when taxonomy expansion fails."""


class TaxonomyExpander:
    """Main class for mining additional taxonomy keywords from review data."""

    def __init__(self, df: pd.DataFrame, language_column: str | None, topics: Sequence[TopicConfig]):
        self.df = df
        self.language_column = language_column
        self.topics = list(topics)
        self.text_column = select_text_column(df)

    def generate(self) -> dict[str, TopicSuggestions]:
        """Generate keyword suggestions for each configured topic."""

        if not self.topics:
            raise TaxonomyExpanderError("No topics with seed keywords were found in taxonomy")

        lang_totals: collections.Counter[str] = collections.Counter()
        global_counts: dict[str, collections.Counter[tuple[str, ...]]] = collections.defaultdict(collections.Counter)
        topic_counts: dict[str, dict[str, collections.Counter[tuple[str, ...]]]] = {
            topic.name: collections.defaultdict(collections.Counter) for topic in self.topics
        }

        for tokens, topic_name, lang in self._iter_windows():
            ngrams = list(self._generate_ngrams(tokens))
            if not ngrams:
                continue
            lang_totals[lang] += 1
            for ngram in ngrams:
                global_counts[lang][ngram] += 1
                topic_counts[topic_name][lang][ngram] += 1

        suggestions: dict[str, TopicSuggestions] = {}
        for topic in self.topics:
            lang_entries: dict[str, list[str]] = {}
            for lang, counts in topic_counts[topic.name].items():
                total_windows = lang_totals.get(lang, 0)
                if total_windows == 0:
                    continue
                scored = self._score_candidates(
                    counts,
                    global_counts[lang],
                    total_windows,
                    topic.existing_terms,
                )
                if not scored:
                    continue
                lang_entries[lang] = scored

            if not lang_entries:
                continue

            aggregated: list[str] = []
            seen: set[str] = set()
            for terms in lang_entries.values():
                for term in terms:
                    if term not in seen:
                        aggregated.append(term)
                        seen.add(term)

            by_language = None
            if len(lang_entries) > 1:
                by_language = lang_entries
            suggestions[topic.name] = TopicSuggestions(add=aggregated[:30], by_language=by_language)

        return suggestions

    def _iter_windows(self) -> Iterator[tuple[list[str], str, str]]:
        lang_col = self.language_column
        default_lang = "unknown"
        seed_token_map = {
            topic.name: [self._tokenize(seed) for seed in topic.seeds] for topic in self.topics
        }

        for _, row in self.df.iterrows():
            raw_text = self._get_text(row)
            if not raw_text:
                continue
            if lang_col:
                raw_lang = row.get(lang_col)
                if isinstance(raw_lang, str):
                    lang = raw_lang.strip().lower() or default_lang
                else:
                    lang = default_lang
            else:
                lang = default_lang
            sentences = self._split_sentences(raw_text)
            for sentence in sentences:
                tokens = self._tokenize(sentence)
                if not tokens:
                    continue
                for topic in self.topics:
                    windows = self._find_windows(tokens, seed_token_map[topic.name])
                    for window in windows:
                        yield window, topic.name, lang

    def _get_text(self, row: pd.Series) -> str:
        value = row.get(self.text_column)
        if isinstance(value, str):
            return value
        return ""

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = text.replace("\n", " ")
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            from nltk.tokenize import sent_tokenize

            return [s for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass
        return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_REGEX.findall(text)]

    def _find_windows(self, tokens: list[str], seed_tokens: list[list[str]]) -> Iterator[list[str]]:
        if not seed_tokens:
            return
        length = len(tokens)
        for seed in seed_tokens:
            if not seed:
                continue
            size = len(seed)
            for idx in range(length - size + 1):
                if tokens[idx : idx + size] == seed:
                    start = max(0, idx - WINDOW_RADIUS)
                    end = min(length, idx + size + WINDOW_RADIUS)
                    yield tokens[start:end]

    def _generate_ngrams(self, tokens: Sequence[str]) -> Iterator[tuple[str, ...]]:
        for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
            if n > len(tokens):
                break
            for idx in range(len(tokens) - n + 1):
                ngram = tuple(tokens[idx : idx + n])
                if not self._is_valid_candidate(ngram):
                    continue
                yield ngram

    def _is_valid_candidate(self, ngram: Sequence[str]) -> bool:
        if not ngram:
            return False
        if all(token in STOPWORDS for token in ngram):
            return False
        if all(token.isdigit() for token in ngram):
            return False
        return True

    def _score_candidates(
        self,
        counts: collections.Counter[tuple[str, ...]],
        global_counts: collections.Counter[tuple[str, ...]],
        total_windows: int,
        existing_terms: set[str],
        limit: int = 30,
    ) -> list[str]:
        scored: list[tuple[float, int, str]] = []
        for ngram, freq in counts.items():
            term = " ".join(ngram)
            if term in existing_terms:
                continue
            global_freq = global_counts.get(ngram, 0)
            score = freq * math.log((total_windows + 1) / (1 + global_freq))
            scored.append((score, freq, term))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [term for _, _, term in scored[:limit]]


def load_taxonomy(path: Path) -> dict:
    if not path.exists():
        raise TaxonomyExpanderError(f"Taxonomy file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TaxonomyExpanderError("Taxonomy file must contain a mapping at the root")
    return data


def extract_topics(taxonomy: dict) -> list[TopicConfig]:
    topics_data = taxonomy.get("topics")
    if isinstance(topics_data, dict):
        items = topics_data.items()
    elif isinstance(topics_data, list):
        items = ((item.get("name"), item) for item in topics_data if isinstance(item, dict))
    else:
        raise TaxonomyExpanderError("Expected 'topics' key in taxonomy")

    configs: list[TopicConfig] = []
    for name, topic_entry in items:
        if not name:
            continue
        seeds = set()
        existing_terms = set()
        if isinstance(topic_entry, dict):
            for key in ("seed_keywords", "seeds", "keywords", "terms", "phrases"):
                values = topic_entry.get(key)
                if isinstance(values, list):
                    normalized = {str(v).strip().lower() for v in values if str(v).strip()}
                    seeds |= normalized
                    existing_terms |= normalized
            additional = topic_entry.get("additional_keywords")
            if isinstance(additional, list):
                normalized = {str(v).strip().lower() for v in additional if str(v).strip()}
                existing_terms |= normalized
        elif isinstance(topic_entry, list):
            normalized = {str(v).strip().lower() for v in topic_entry if str(v).strip()}
            seeds |= normalized
            existing_terms |= normalized
        if seeds:
            configs.append(TopicConfig(name=name, seeds=seeds, existing_terms=existing_terms))
    return configs


def select_language_column(df: pd.DataFrame) -> str | None:
    for candidate in ("lang", "language", "locale"):
        if candidate in df.columns:
            return candidate
    return None


def select_text_column(df: pd.DataFrame) -> str:
    for candidate in ("review_text", "text", "body", "content", "review"):
        if candidate in df.columns:
            return candidate
    raise TaxonomyExpanderError(
        "Review text column not found. Expected one of: review_text, text, body, content, review"
    )


def write_suggestions(path: Path, suggestions: dict[str, TopicSuggestions]) -> None:
    payload: dict[str, object] = {"topics": {}}
    for topic, suggestion in suggestions.items():
        entry: dict[str, object] = {"add": suggestion.add}
        if suggestion.by_language:
            entry["by_language"] = suggestion.by_language
        payload["topics"][topic] = entry
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, allow_unicode=True)


def apply_to_taxonomy(taxonomy_path: Path, suggestions: dict[str, TopicSuggestions]) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    topics = taxonomy.get("topics")
    if not isinstance(topics, dict):
        raise TaxonomyExpanderError("Cannot apply suggestions: taxonomy topics must be a mapping")

    for topic_name, suggestion in suggestions.items():
        if topic_name not in topics:
            continue
        entry = topics[topic_name]
        terms_to_add = suggestion.add
        if not terms_to_add:
            continue
        if isinstance(entry, list):
            current = {str(v).strip().lower(): str(v).strip() for v in entry if str(v).strip()}
            for term in terms_to_add:
                normalized = term.lower()
                if normalized not in current:
                    current[normalized] = term
            updated = sorted(current.values(), key=str.lower)
            topics[topic_name] = updated
        elif isinstance(entry, dict):
            target_key = _resolve_target_list_key(entry)
            current_list = entry.setdefault(target_key, [])
            if not isinstance(current_list, list):
                raise TaxonomyExpanderError(
                    f"Cannot apply suggestions: '{target_key}' is not a list for topic '{topic_name}'"
                )
            current = {str(v).strip().lower(): str(v).strip() for v in current_list if str(v).strip()}
            for term in terms_to_add:
                normalized = term.lower()
                if normalized not in current:
                    current[normalized] = term
            entry[target_key] = sorted(current.values(), key=str.lower)
        else:
            raise TaxonomyExpanderError(
                f"Cannot apply suggestions: topic '{topic_name}' must be list or dict, got {type(entry)!r}"
            )

    with taxonomy_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(taxonomy, fh, sort_keys=True, allow_unicode=True)


def _resolve_target_list_key(entry: dict) -> str:
    for key in ("keywords", "terms", "phrases", "seed_keywords", "seeds"):
        value = entry.get(key)
        if isinstance(value, list):
            return key
    entry.setdefault("keywords", [])
    return "keywords"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand taxonomy keywords from review data")
    parser.add_argument("--in-parquet", required=True, dest="in_parquet", help="Path to filtered reviews parquet")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy YAML")
    parser.add_argument("--out", required=True, help="Path to suggestions YAML output")
    parser.add_argument("--apply", action="store_true", help="Merge suggested keywords into taxonomy")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    parquet_path = Path(args.in_parquet)
    taxonomy_path = Path(args.taxonomy)
    out_path = Path(args.out)

    df = pd.read_parquet(parquet_path)
    taxonomy = load_taxonomy(taxonomy_path)
    topics = extract_topics(taxonomy)
    if not topics:
        raise TaxonomyExpanderError("No topics with seed keywords found in taxonomy")

    language_column = select_language_column(df)
    expander = TaxonomyExpander(df, language_column, topics)
    suggestions = expander.generate()

    write_suggestions(out_path, suggestions)

    if args.apply:
        apply_to_taxonomy(taxonomy_path, suggestions)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
