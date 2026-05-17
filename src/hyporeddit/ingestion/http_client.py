"""Polite HTTP client with rate limiting, exponential backoff, and circuit breaker.

All outbound HTTP requests from the ingestion pipeline go through this class.
"""

import time
from collections import deque
from typing import Any

import httpx
from loguru import logger

from hyporeddit.config import settings


class IngestionError(Exception):
    """Raised when max retries are exhausted for an HTTP request."""


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open (too many recent errors)."""


class PolitHttpClient:
    """Rate-limited, retrying HTTP client for Reddit's public JSON endpoints."""

    _WINDOW_SIZE = 20

    def __init__(self) -> None:
        self._http = httpx.Client(
            headers={"User-Agent": "python:hyporeddit-validator:v0.1"},
            timeout=30.0,
            follow_redirects=True,
        )
        self._backoff_sequence: list[int] = list(settings.backoff_sequence)
        self._max_retries: int = settings.max_retries
        self._request_delay: float = settings.request_delay_seconds
        self._error_threshold: float = settings.circuit_breaker_threshold
        self._circuit_pause: int = settings.circuit_breaker_pause_seconds
        self._request_window: deque[bool] = deque(maxlen=self._WINDOW_SIZE)
        self._circuit_open_until: float = 0.0

    def _is_circuit_open(self) -> bool:
        if time.time() < self._circuit_open_until:
            return True
        if len(self._request_window) < self._WINDOW_SIZE:
            return False
        error_rate = sum(self._request_window) / len(self._request_window)
        return error_rate > self._error_threshold

    def _record(self, is_error: bool) -> None:
        self._request_window.append(is_error)
        if self._is_circuit_open() and self._circuit_open_until <= time.time():
            self._circuit_open_until = time.time() + self._circuit_pause
            logger.critical(
                "Circuit breaker OPEN — error rate exceeded {}%. "
                "Pausing for {} seconds.",
                int(self._error_threshold * 100),
                self._circuit_pause,
            )

    def get(self, url: str, **kwargs: Any) -> Any:
        """GET a URL, parse JSON, and return the payload.

        Retries on 429/503 with exponential backoff.
        Raises IngestionError after max retries exhausted.
        Raises CircuitBreakerOpen if error rate is too high.
        """
        if self._is_circuit_open():
            raise CircuitBreakerOpen(
                "Circuit breaker is open — too many recent errors. "
                f"Resuming in {int(self._circuit_open_until - time.time())}s."
            )

        last_exc: Exception | None = None
        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                delay = self._backoff_sequence[min(attempt - 1, len(self._backoff_sequence) - 1)]
                logger.warning(
                    "Retry {}/{} for {} — waiting {}s",
                    attempt, self._max_retries, url, delay,
                )
                time.sleep(delay)
            else:
                if self._request_delay > 0:
                    time.sleep(self._request_delay)

            logger.debug("GET {} (attempt {})", url, attempt + 1)
            try:
                resp = self._http.get(url, **kwargs)
                if resp.status_code in (429, 503):
                    self._record(True)
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                    continue
                resp.raise_for_status()
                self._record(False)
                return resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                self._record(True)
                last_exc = exc

        raise IngestionError(
            f"Max retries ({self._max_retries}) exhausted for {url}"
        ) from last_exc

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PolitHttpClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


if __name__ == "__main__":
    # Run: python -m hyporeddit.ingestion.http_client
    # Env: none required
    client = PolitHttpClient()
    client._request_delay = 0.0  # no delay for smoke test
    from hyporeddit.config import settings
    subreddit = settings.reddit_subreddit
    result = client.get(f"https://www.reddit.com/r/{subreddit}/new.json?limit=1")
    posts = result["data"]["children"]
    logger.info("Fetched {} post(s) from r/{}", len(posts), subreddit)
    if posts:
        logger.info("First post title: {}", posts[0]["data"]["title"])
    client.close()
