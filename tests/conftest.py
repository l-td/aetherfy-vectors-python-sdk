"""
Pytest configuration and fixtures for Aetherfy Vectors SDK tests.

Provides common test fixtures and configuration for all test modules.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import Collection, VectorConfig, DistanceMetric


@pytest.fixture
def api_key():
    """Test API key fixture."""
    return "afy_test_1234567890abcdef1234"


@pytest.fixture
def test_endpoint():
    """Test endpoint URL fixture."""
    return "https://test-api.aetherfy.com"


@pytest.fixture
def client(api_key, test_endpoint, mock_requests):
    """AetherfyVectorsClient fixture with test configuration.

    Note: mock_requests is a dependency to ensure requests are mocked before client init.
    """
    return AetherfyVectorsClient(
        api_key=api_key,
        endpoint=test_endpoint,
        timeout=10.0
    )


@pytest.fixture
def mock_requests():
    """Mock requests module for testing."""
    import requests
    with patch('aetherfy_vectors.client.requests') as mock:
        # Ensure exception classes are properly set up
        mock.Timeout = requests.Timeout
        mock.RequestException = requests.RequestException
        mock.ConnectionError = requests.ConnectionError

        # Create a mock session that behaves like the patched requests module
        mock_session = Mock()

        # Wrap the request method to merge headers properly like a real session does
        def session_request_wrapper(*args, **kwargs):
            # Merge session headers with request headers
            merged_headers = mock_session.headers.copy()
            if 'headers' in kwargs and kwargs['headers']:
                merged_headers.update(kwargs['headers'])
            kwargs['headers'] = merged_headers
            return mock.request(*args, **kwargs)

        mock_session.request = session_request_wrapper
        mock_session.get = mock.get
        mock_session.post = mock.post
        mock_session.put = mock.put
        mock_session.delete = mock.delete

        # Mock headers with a real dict so .update() works
        mock_session.headers = {}
        mock_session.close = Mock()

        # Mock mount method for HTTPAdapter mounting
        mock_session.mount = Mock()

        # Mock Session() to return our mock session
        mock.Session.return_value = mock_session

        yield mock


@pytest.fixture
def sample_collection():
    """Sample collection fixture."""
    return Collection(
        name="test_collection",
        config=VectorConfig(size=128, distance=DistanceMetric.COSINE),
        points_count=100,
        status="green"
    )


@pytest.fixture
def sample_points():
    """Sample points data fixture."""
    return [
        {
            "id": "point_1",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "payload": {"category": "test", "value": 42}
        },
        {
            "id": "point_2", 
            "vector": [0.5, 0.6, 0.7, 0.8],
            "payload": {"category": "example", "value": 84}
        }
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results fixture."""
    return [
        {
            "id": "point_1",
            "score": 0.95,
            "payload": {"category": "test", "value": 42},
            "vector": [0.1, 0.2, 0.3, 0.4]
        },
        {
            "id": "point_2",
            "score": 0.87,
            "payload": {"category": "example", "value": 84},
            "vector": [0.5, 0.6, 0.7, 0.8]
        }
    ]


@pytest.fixture
def mock_successful_response():
    """Mock successful HTTP response."""
    def _create_response(data: Dict[str, Any], status_code: int = 200):
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data
        mock_response.content = True
        return mock_response
    return _create_response


@pytest.fixture
def mock_error_response():
    """Mock error HTTP response."""
    def _create_error_response(
        message: str = "Test error",
        status_code: int = 400,
        error_code: str = "test_error",
        request_id: str = "req_123"
    ):
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = {
            "message": message,
            "error_code": error_code,
            "request_id": request_id,
            "details": {}
        }
        mock_response.content = True
        return mock_response
    return _create_error_response


@pytest.fixture
def sample_performance_analytics():
    """Sample performance analytics data."""
    return {
        "cache_hit_rate": 0.85,
        "avg_latency_ms": 23.5,
        "requests_per_second": 150.0,
        "active_regions": ["us-east-1", "eu-central-1", "ap-southeast-1"],
        "region_performance": {
            "us-east-1": {"latency_ms": 20.1, "requests_per_second": 60.0},
            "eu-central-1": {"latency_ms": 25.3, "requests_per_second": 45.0},
            "ap-southeast-1": {"latency_ms": 28.7, "requests_per_second": 45.0}
        },
        "total_requests": 129600,
        "error_rate": 0.002
    }


@pytest.fixture
def sample_collection_analytics():
    """Sample collection analytics data."""
    return {
        "collection_name": "test_collection",
        "total_points": 1000,
        "search_requests": 500,
        "avg_search_latency_ms": 18.5,
        "cache_hit_rate": 0.92,
        "top_regions": ["us-east-1", "eu-central-1"],
        "storage_size_mb": 45.2
    }


@pytest.fixture
def sample_usage_stats():
    """Sample usage statistics data."""
    return {
        "current_collections": 5,
        "max_collections": 10,
        "current_points": 50000,
        "max_points": 100000,
        "requests_this_month": 25000,
        "max_requests_per_month": 100000,
        "storage_used_mb": 250.5,
        "max_storage_mb": 1000.0,
        "plan_name": "Professional"
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    import os
    original_env = os.environ.copy()
    
    # Remove any API key environment variables
    for key in ["AETHERFY_API_KEY", "AETHERFY_VECTORS_API_KEY"]:
        if key in os.environ:
            del os.environ[key]
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


class MockAnalyticsClient:
    """Mock analytics client for testing."""
    
    def __init__(self, base_url: str, auth_headers: Dict[str, str], timeout: float = 30.0):
        self.base_url = base_url
        self.auth_headers = auth_headers
        self.timeout = timeout
    
    def get_performance_analytics(self, time_range: str = "24h", region: str = None):
        from aetherfy_vectors.models import PerformanceAnalytics
        return PerformanceAnalytics.from_dict({
            "cache_hit_rate": 0.85,
            "avg_latency_ms": 23.5,
            "requests_per_second": 150.0,
            "active_regions": ["us-east-1", "eu-central-1"],
            "region_performance": {}
        })
    
    def get_collection_analytics(self, collection_name: str, time_range: str = "24h"):
        from aetherfy_vectors.models import CollectionAnalytics
        return CollectionAnalytics.from_dict({
            "collection_name": collection_name,
            "total_points": 1000,
            "search_requests": 500,
            "avg_search_latency_ms": 18.5,
            "cache_hit_rate": 0.92,
            "top_regions": ["us-east-1"]
        })
    
    def get_usage_stats(self):
        from aetherfy_vectors.models import UsageStats
        return UsageStats.from_dict({
            "current_collections": 5,
            "max_collections": 10,
            "current_points": 50000,
            "max_points": 100000,
            "requests_this_month": 25000,
            "max_requests_per_month": 100000,
            "storage_used_mb": 250.5,
            "max_storage_mb": 1000.0,
            "plan_name": "Professional"
        })


@pytest.fixture
def mock_analytics_client():
    """Mock analytics client fixture."""
    return MockAnalyticsClient