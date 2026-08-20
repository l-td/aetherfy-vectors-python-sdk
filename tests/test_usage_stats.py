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
"""

import pytest
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
        assert stats.current_collections == 5
        assert stats.max_collections == 10
        assert stats.plan_name == "Professional"
        assert stats.collections_usage_percent == 50.0
        assert stats.points_usage_percent == 50.0

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
    """UsageStats.from_dict and its derived percentages."""

    def test_from_dict(self, sample_usage_stats):
        stats = UsageStats.from_dict(sample_usage_stats)

        assert stats.current_collections == 5
        assert stats.max_collections == 10
        assert stats.plan_name == "Professional"

        assert stats.collections_usage_percent == 50.0
        assert stats.points_usage_percent == 50.0
        assert stats.requests_usage_percent == 25.0
        assert stats.storage_usage_percent == 25.05

    def test_percentage_calculations(self):
        stats = UsageStats.from_dict(
            {
                "current_collections": 3,
                "max_collections": 10,
                "current_points": 75000,
                "max_points": 100000,
                "requests_this_month": 30000,
                "max_requests_per_month": 50000,
                "storage_used_mb": 400.0,
                "max_storage_mb": 800.0,
                "plan_name": "Starter",
            }
        )

        assert stats.collections_usage_percent == 30.0
        assert stats.points_usage_percent == 75.0
        assert stats.requests_usage_percent == 60.0
        assert stats.storage_usage_percent == 50.0
