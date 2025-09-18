#!/usr/bin/env python3
"""Command line tool for exporting Steam reviews to CSV."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests
from requests import Response


API_URL_TEMPLATE = "https://store.steampowered.com/appreviews/{app_id}"
DEFAULT_REQUEST_PARAMS = {
    "json": 1,
    "filter": "recent",
    "review_type": "all",
    "purchase_type": "all",
    "num_per_page": 100,
}


@dataclass
class Review:
    """Simple representation of a Steam review record."""

    review_id: str
    language: str
    review_text: str
    timestamp_created: int
    timestamp_updated: int
    voted_up: bool
    votes_up: int
    votes_funny: int
    weighted_vote_score: str
    steam_purchase: bool
    received_for_free: bool
    written_during_early_access: bool

    @classmethod
    def from_api(cls, data: dict) -> "Review":
        return cls(
            review_id=str(data.get("recommendationid", "")),
            language=data.get("language", ""),
            review_text=data.get("review", ""),
            timestamp_created=int(data.get("timestamp_created", 0)),
            timestamp_updated=int(data.get("timestamp_updated", 0)),
            voted_up=bool(data.get("voted_up", False)),
            votes_up=int(data.get("votes_up", 0)),
            votes_funny=int(data.get("votes_funny", 0)),
            weighted_vote_score=str(data.get("weighted_vote_score", "")),
            steam_purchase=bool(data.get("steam_purchase", False)),
            received_for_free=bool(data.get("received_for_free", False)),
            written_during_early_access=bool(data.get("written_during_early_access", False)),
        )

    def to_row(self) -> List[str]:
        return [
            self.review_id,
            self.language,
            self.review_text,
            str(self.timestamp_created),
            str(self.timestamp_updated),
            "1" if self.voted_up else "0",
            str(self.votes_up),
            str(self.votes_funny),
            self.weighted_vote_score,
            "1" if self.steam_purchase else "0",
            "1" if self.received_for_free else "0",
            "1" if self.written_during_early_access else "0",
        ]


class SteamReviewExtractor:
    """Fetches reviews from the Steam Store API."""

    def __init__(
        self,
        app_id: str,
        language: str = "english",
        max_reviews: Optional[int] = None,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        request_timeout: float = 30.0,
    ) -> None:
        self.app_id = app_id
        self.language = language
        self.max_reviews = max_reviews
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.request_timeout = request_timeout

    def fetch_reviews(self) -> Iterable[Review]:
        cursor = "*"
        fetched = 0

        while True:
            batch, cursor = self._fetch_batch(cursor)
            if not batch:
                break

            for review in batch:
                yield review
                fetched += 1
                if self.max_reviews is not None and fetched >= self.max_reviews:
                    return

            if cursor is None:
                break

    def _fetch_batch(self, cursor: str) -> tuple[List[Review], Optional[str]]:
        params = dict(DEFAULT_REQUEST_PARAMS)
        params.update({
            "language": self.language,
            "cursor": cursor,
        })

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.get(
                    API_URL_TEMPLATE.format(app_id=self.app_id),
                    params=params,
                    timeout=self.request_timeout,
                )
                if response.status_code == 429:
                    self._handle_retry_delay(attempt)
                    continue
                response.raise_for_status()
                return self._parse_response(response)
            except requests.RequestException:
                if attempt == self.retries:
                    raise
                self._handle_retry_delay(attempt)

        return [], None

    def _handle_retry_delay(self, attempt: int) -> None:
        delay = self.backoff_seconds * attempt
        time.sleep(delay)

    def _parse_response(self, response: Response) -> tuple[List[Review], Optional[str]]:
        payload = response.json()
        success = payload.get("success")
        if success != 1:
            return [], None

        review_dicts = payload.get("reviews", [])
        reviews = [Review.from_api(review) for review in review_dicts]

        next_cursor = payload.get("cursor")
        if not review_dicts or not next_cursor:
            next_cursor = None
        return reviews, next_cursor


def write_reviews_to_csv(reviews: Iterable[Review], output_path: str) -> None:
    header = [
        "review_id",
        "language",
        "review_text",
        "timestamp_created",
        "timestamp_updated",
        "voted_up",
        "votes_up",
        "votes_funny",
        "weighted_vote_score",
        "steam_purchase",
        "received_for_free",
        "written_during_early_access",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for review in reviews:
            writer.writerow(review.to_row())


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Steam reviews to CSV.")
    parser.add_argument("app_id", help="Steam App ID for the game")
    parser.add_argument(
        "-o",
        "--output",
        default="reviews.csv",
        help="Path to output CSV file (default: reviews.csv)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="english",
        help="Review language to request from the API (default: english)",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Maximum number of reviews to fetch (default: all available)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of times to retry failed requests (default: 3)",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.0,
        help="Base backoff seconds between retries (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30.0)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    extractor = SteamReviewExtractor(
        app_id=args.app_id,
        language=args.language,
        max_reviews=args.max_reviews,
        retries=args.retries,
        backoff_seconds=args.backoff,
        request_timeout=args.timeout,
    )

    try:
        reviews = list(extractor.fetch_reviews())
    except requests.RequestException as exc:
        print(f"Failed to fetch reviews: {exc}", file=sys.stderr)
        return 1

    write_reviews_to_csv(reviews, args.output)
    print(f"Fetched {len(reviews)} reviews for app {args.app_id}.")
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
