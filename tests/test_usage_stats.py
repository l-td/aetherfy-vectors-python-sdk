"""Tests for usage statistics — the one surviving analytics surface.

This file is what remains of test_analytics.py. AnalyticsClient was deleted
along with every method it carried except this one: get_performance_analytics
reported a region_performance the backend synthesised rather than measured, and
get_region_performance / get_cache_analytics called routes that do not exist.
GET /api/v1/analytics/usage is the only analytics endpoint backed by real data
(the backend reads Postgres for it), so it is the only one that survived.

The patch target moved with the code. These tests used to patch
`aetherfy_vectors.analytics.requests.get`; the method now lives on
AetherfyVectorsClient and issues its request through the client's own pooled
`self.session`, so the session is what gets patched. Patching the module-level
`requests.get` here would pass while asserting nothing — the call would never
go through it.

WHAT THIS FILE IS NOT. Every payload here is mocked, so nothing below can tell
you the endpoint still serves this shape — that is exactly how UsageStats spent
its whole life describing nine fields the backend never sent. The authoritative
pin is the live call in aetherfy-e2e-tests tests/sdk/test_usage_stats_sdk.py; the
`sample_usage_stats` fixture is a copy of that truth and must only ever change
together with it.
"""

import pytest
from dataclasses import fields
from unittest.mock import Mock, patch
import requests

from aetherfy_vectors.models import UsageStats
from aetherfy_vectors.exceptions import AetherfyVectorsException


def _response(status_code, json_body, content=True):
    r = Mock()
    r.status_code = status_code
    r.json.return_value = json_body
    r.content = content
    return r


class TestGetUsageStats:
    """AetherfyVectorsClient.get_usage_stats over the wire."""

    def test_success(self, client, sample_usage_stats):
        with patch.object(
            client.session, "get", return_value=_response(200, sample_usage_stats)
        ) as mock_get:
            stats = client.get_usage_stats()

        assert isinstance(stats, UsageStats)
        assert stats.storage_bytes_used == 268_435_456
        assert stats.storage_limit_bytes == 1_073_741_824
        assert stats.collections_count == 5
        assert stats.collections_limit == 10
        assert stats.tier == "professional"
        assert stats.active_regions == ["us-east-1", "eu-central-1"]
        assert stats.usage_percentage == 25

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        # The endpoint, pinned: this is the contract the backend keeps serving.
        assert "analytics/usage" in args[0]
        # Sent with the client's auth headers and timeout, not bare defaults.
        assert kwargs["headers"] == client.auth_headers
        assert kwargs["timeout"] == client.timeout

    def test_non_200_raises(self, client):
        with patch.object(
            client.session,
            "get",
            return_value=_response(403, {"error": {"code": "ACCOUNT_SUSPENDED"}}),
        ):
            with pytest.raises(AetherfyVectorsException):
                client.get_usage_stats()

    def test_empty_response_body_raises(self, client):
        """A 500 with no body must still raise, not crash on .json()."""
        with patch.object(
            client.session, "get", return_value=_response(500, {}, content=None)
        ):
            with pytest.raises(AetherfyVectorsException):
                client.get_usage_stats()

    def test_request_exception_is_wrapped(self, client):
        with patch.object(
            client.session, "get", side_effect=requests.RequestException("Network error")
        ):
            with pytest.raises(AetherfyVectorsException) as exc_info:
                client.get_usage_stats()

        assert "Failed to retrieve usage statistics" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)


class TestUsageStatsModel:
    """UsageStats.from_dict against the real wire body."""

    def test_from_dict(self, sample_usage_stats):
        stats = UsageStats.from_dict(sample_usage_stats)

        assert stats.storage_bytes_used == 268_435_456
        assert stats.storage_limit_bytes == 1_073_741_824
        assert stats.collections_count == 5
        assert stats.collections_limit == 10
        assert stats.tier == "professional"
        assert stats.active_regions == ["us-east-1", "eu-central-1"]
        assert stats.usage_percentage == 25

    def test_unlimited_tier_nulls_BOTH_limits(self):
        """An unlimited tier serves `null` for both limit fields.

        One sentinel, not two. `customerStore` represents "no limit" as the
        STRING 'unlimited' inside vectordb, but that is an internal
        limits-vocabulary convention and the endpoint normalises both fields
        to null before they reach the wire — pinned on the server side by
        vectordb tests/unit/analyticsMetricsRead.test.js, "an unlimited tier
        reports BOTH limits as null, in one vocabulary".

        This case previously used `collections_limit: -1`, a payload the
        backend has never sent. Inventing a sentinel to test against is the
        exact defect this file exists to close.
        """
        stats = UsageStats.from_dict(
            {
                "storage_bytes_used": 42,
                "storage_limit_bytes": None,
                "collections_count": 3,
                "collections_limit": None,
                "tier": "enterprise",
                "active_regions": [],
                "usage_percentage": 0,
            }
        )

        assert stats.storage_limit_bytes is None
        assert stats.collections_limit is None
        assert stats.active_regions == []
        assert stats.usage_percentage == 0

    def test_no_request_count_is_required_or_exposed(self, sample_usage_stats):
        """The removed hourly/monthly request counters must stay removed.

        `requests_this_hour` was deleted from the endpoint in 2026-09 (it read
        a Redis key nothing had ever written, so every customer was told 0),
        and `requests_this_month` never existed at all — it was one of the nine
        invented fields this model used to declare. Neither may come back as a
        parse requirement or as an attribute: a body carrying only the seven
        real fields must parse, and the parsed object must not answer to
        either name.
        """
        assert "requests_this_hour" not in sample_usage_stats
        assert "requests_this_month" not in sample_usage_stats

        stats = UsageStats.from_dict(sample_usage_stats)

        assert not hasattr(stats, "requests_this_hour")
        assert not hasattr(stats, "requests_this_month")

    def test_field_set_is_exactly_the_wire_body(self, sample_usage_stats):
        """No field may exist that the endpoint does not serve.

        This is the assertion that would have caught the original defect: the
        model's fields and the wire body's keys are the same set, so an
        invented field cannot be added without a wire body to justify it.
        """
        assert {f.name for f in fields(UsageStats)} == set(sample_usage_stats)
