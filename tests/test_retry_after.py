"""``Retry-After`` on a 503 (RESOURCE_BUSY, SERVICE_UNAVAILABLE).

A retried operation waits the seconds the header names: capped, never past
the call's own deadline. A missing or unreadable header falls back to the
exponential backoff. Time is a fake clock: nothing really sleeps.
"""

from email.utils import parsedate_to_datetime
from unittest.mock import Mock, patch

import pytest

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.exceptions import ServiceUnavailableError
from aetherfy_vectors.retry_policy import REPEATABLE
from aetherfy_vectors.utils import (
    parse_error_response,
    parse_retry_after,
    retry_with_backoff,
)


class Clock:
    """A fake monotonic clock; sleeping advances it and is recorded."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    c = Clock()
    with patch("aetherfy_vectors.utils.time.monotonic", c.monotonic), patch(
        "aetherfy_vectors.utils.time.sleep", c.sleep
    ):
        yield c


def busy(retry_after):
    return ServiceUnavailableError("busy", retry_after=retry_after, status_code=503)


def failing_then_ok(*errors):
    errors = list(errors)

    def func():
        if errors:
            raise errors.pop(0)
        return "ok"

    return func


# -- reading the header ------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("7", 7.0),
        (" 2.5 ", 2.5),
        ("0", 0.0),
        ("", None),
        ("soon", None),
        ("-3", None),
        ("nan", None),
        ("inf", None),
        (None, None),
        (7, None),
    ],
)
def test_the_header_is_read_as_seconds_or_not_at_all(value, expected):
    assert parse_retry_after(value) == expected


def test_an_http_date_is_the_seconds_until_it():
    date = "Wed, 21 Oct 2026 07:28:30 GMT"
    target = parsedate_to_datetime(date).timestamp()
    assert parse_retry_after(date, now=target - 12) == 12.0
    assert parse_retry_after(date, now=target + 50) == 0.0


def test_the_503_error_carries_the_header():
    err = parse_error_response(
        {"error": {"code": "RESOURCE_BUSY", "message": "busy"}},
        503,
        {"Retry-After": "4"},
    )
    assert isinstance(err, ServiceUnavailableError)
    assert err.retry_after == 4.0


def test_a_503_without_the_header_has_none():
    err = parse_error_response(
        {"error": {"code": "RESOURCE_BUSY", "message": "x"}}, 503, {}
    )
    assert err.retry_after is None
    assert parse_error_response({"message": "x"}, 503).retry_after is None


def test_a_header_object_that_is_not_a_mapping_is_ignored():
    assert parse_error_response({"message": "x"}, 503, Mock()).retry_after is None


# -- waiting it --------------------------------------------------------------


def test_the_wait_is_the_headers_value_not_the_backoff(clock):
    assert retry_with_backoff(failing_then_ok(busy(7)), REPEATABLE) == "ok"
    assert clock.slept == [7.0]


def test_each_retry_waits_its_own_ask(clock):
    retry_with_backoff(failing_then_ok(busy(2), busy(5)), REPEATABLE)
    assert clock.slept == [2.0, 5.0]


def test_the_wait_is_capped(clock):
    retry_with_backoff(failing_then_ok(busy(3600)), REPEATABLE, max_delay=30.0)
    assert clock.slept == [30.0]


def test_a_wait_past_the_deadline_is_not_taken(clock):
    deadline = clock.now + 5
    with pytest.raises(ServiceUnavailableError):
        retry_with_backoff(failing_then_ok(busy(10)), REPEATABLE, deadline=deadline)
    assert clock.slept == []


def test_a_wait_that_fits_the_deadline_is_taken(clock):
    deadline = clock.now + 20
    assert (
        retry_with_backoff(failing_then_ok(busy(10)), REPEATABLE, deadline=deadline)
        == "ok"
    )
    assert clock.slept == [10.0]


def test_a_missing_header_falls_back_to_the_backoff(clock):
    retry_with_backoff(failing_then_ok(busy(None)), REPEATABLE, base_delay=4.0)
    assert len(clock.slept) == 1
    assert 2.0 <= clock.slept[0] <= 4.0  # base 4 s, jittered to 50-100%


def test_a_zero_wait_retries_at_once(clock):
    retry_with_backoff(failing_then_ok(busy(0)), REPEATABLE)
    assert clock.slept == [0.0]


# -- through the client ------------------------------------------------------


def _response(status, payload, headers=None):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    r.content = True
    r.headers = headers or {}
    return r


def test_the_clients_retry_waits_the_header(mock_requests, clock):
    client = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="home",
    )
    mock_requests.request.side_effect = [
        _response(
            503,
            {"error": {"code": "RESOURCE_BUSY", "message": "busy"}},
            {"Retry-After": "6"},
        ),
        _response(200, {"result": {"count": 4}}),
    ]
    assert client.count("docs") == 4
    assert clock.slept == [6.0]
    assert mock_requests.request.call_count == 2
