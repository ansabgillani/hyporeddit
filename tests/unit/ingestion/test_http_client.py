"""Unit tests for ingestion/http_client.py.

No network calls — httpx is mocked at the usage site.
All tests complete in under 1 second (backoff delays are overridden in config).
"""

from unittest.mock import MagicMock, patch

import pytest

from hyporeddit.ingestion.http_client import CircuitBreakerOpen, IngestionError, PolitHttpClient


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {"data": {}}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


def make_client(
    backoff_sequence: list[int] | None = None,
    max_retries: int = 3,
    request_delay: float = 0.0,
) -> PolitHttpClient:
    """Create a client with fast backoff for unit tests."""
    client = PolitHttpClient()
    client._backoff_sequence = backoff_sequence or [0, 0, 0]
    client._max_retries = max_retries
    client._request_delay = request_delay
    return client


# ---------------------------------------------------------------------------
# Basic request tests
# ---------------------------------------------------------------------------

def test_get_success_returns_json() -> None:
    client = make_client()
    mock_resp = make_response(200, {"data": {"children": []}})
    with patch.object(client._http, "get", return_value=mock_resp):
        result = client.get("https://example.com/test.json")
    assert result == {"data": {"children": []}}


def test_user_agent_header_set() -> None:
    client = make_client()
    assert "hyporeddit-validator" in client._http.headers["user-agent"]


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

def test_retries_on_429_then_succeeds() -> None:
    client = make_client()
    import httpx

    fail_resp = make_response(429)
    ok_resp = make_response(200, {"ok": True})

    calls = [fail_resp, ok_resp]
    with patch.object(client._http, "get", side_effect=calls):
        result = client.get("https://example.com/test.json")
    assert result == {"ok": True}


def test_retries_on_503_then_succeeds() -> None:
    client = make_client()
    fail_resp = make_response(503)
    ok_resp = make_response(200, {"ok": True})

    with patch.object(client._http, "get", side_effect=[fail_resp, ok_resp]):
        result = client.get("https://example.com/test.json")
    assert result == {"ok": True}


def test_raises_ingestion_error_after_max_retries() -> None:
    client = make_client(max_retries=2)
    fail_resp = make_response(429)

    with patch.object(client._http, "get", return_value=fail_resp):
        with pytest.raises(IngestionError):
            client.get("https://example.com/test.json")


def test_retry_count_matches_max_retries() -> None:
    client = make_client(max_retries=3)
    fail_resp = make_response(429)

    with patch.object(client._http, "get", return_value=fail_resp) as mock_get:
        with pytest.raises(IngestionError):
            client.get("https://example.com/test.json")
    # initial attempt + max_retries retries
    assert mock_get.call_count == 1 + 3


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------

def test_circuit_breaker_opens_after_threshold() -> None:
    """50%+ error rate in 20-request window should open the circuit."""
    client = make_client()
    # Manually set up the sliding window with >50% errors (11 out of 20)
    client._request_window = [False] * 9 + [True] * 11  # 11 errors out of 20

    assert client._is_circuit_open() is True


def test_circuit_breaker_closed_below_threshold() -> None:
    client = make_client()
    client._request_window = [False] * 15 + [True] * 5  # 5 errors out of 20 = 25%

    assert client._is_circuit_open() is False


def test_circuit_breaker_raises_on_open_circuit() -> None:
    client = make_client()
    client._request_window = [True] * 20  # all errors

    with pytest.raises(CircuitBreakerOpen):
        client.get("https://example.com/test.json")


def test_request_window_tracks_successes_and_failures() -> None:
    client = make_client()
    ok_resp = make_response(200, {"ok": True})

    with patch.object(client._http, "get", return_value=ok_resp):
        client.get("https://example.com/test.json")

    # Window should have recorded a success (False = no error)
    assert client._request_window[-1] is False


def test_request_window_capped_at_20() -> None:
    client = make_client()
    ok_resp = make_response(200, {"ok": True})

    with patch.object(client._http, "get", return_value=ok_resp):
        for _ in range(25):
            client.get("https://example.com/test.json")

    assert len(client._request_window) == 20


# ---------------------------------------------------------------------------
# Politeness delay tests
# ---------------------------------------------------------------------------

def test_default_request_delay_is_1_second() -> None:
    """Default politeness delay must be 1.0 s (1000 ms) between Reddit API calls."""
    from hyporeddit.config import Settings
    assert Settings().request_delay_seconds == 1.0


def test_sleeps_request_delay_before_each_request() -> None:
    """PolitHttpClient sleeps _request_delay seconds before each non-retry GET."""
    client = make_client(request_delay=1.0)
    ok_resp = make_response(200, {"ok": True})

    with patch("hyporeddit.ingestion.http_client.time.sleep") as mock_sleep, \
         patch.object(client._http, "get", return_value=ok_resp):
        client.get("https://example.com/test.json")

    mock_sleep.assert_called_once_with(1.0)
