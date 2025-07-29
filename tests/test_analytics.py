"""
Tests for analytics functionality.

Tests analytics data retrieval, performance metrics, and usage statistics.
"""

import pytest
from unittest.mock import Mock, patch
import requests

from aetherfy_vectors.analytics import AnalyticsClient
from aetherfy_vectors.models import PerformanceAnalytics, CollectionAnalytics, UsageStats
from aetherfy_vectors.exceptions import AetherfyVectorsException


class TestAnalyticsClient:
    """Test AnalyticsClient functionality."""
    
    @pytest.fixture
    def analytics_client(self):
        """Analytics client fixture."""
        auth_headers = {"Authorization": "Bearer test_key", "X-API-Key": "test_key"}
        return AnalyticsClient("https://test-api.aetherfy.com", auth_headers, timeout=10.0)
    
    def test_analytics_client_initialization(self, analytics_client):
        """Test analytics client initialization."""
        assert analytics_client.base_url == "https://test-api.aetherfy.com"
        assert analytics_client.timeout == 10.0
        assert "Authorization" in analytics_client.auth_headers
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_get_performance_analytics_success(self, mock_get, analytics_client, sample_performance_analytics):
        """Test successful performance analytics retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_performance_analytics
        mock_get.return_value = mock_response
        
        analytics = analytics_client.get_performance_analytics()
        
        assert isinstance(analytics, PerformanceAnalytics)
        assert analytics.cache_hit_rate == 0.85
        assert analytics.avg_latency_ms == 23.5
        assert analytics.requests_per_second == 150.0
        assert len(analytics.active_regions) == 3
        
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "analytics/performance" in args[0]
        assert kwargs["params"]["time_range"] == "24h"
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_get_performance_analytics_with_region(self, mock_get, analytics_client, sample_performance_analytics):
        """Test performance analytics retrieval with region filter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_performance_analytics
        mock_get.return_value = mock_response
        
        analytics = analytics_client.get_performance_analytics(time_range="7d", region="us-east-1")
        
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["time_range"] == "7d"
        assert kwargs["params"]["region"] == "us-east-1"
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_get_collection_analytics_success(self, mock_get, analytics_client, sample_collection_analytics):
        """Test successful collection analytics retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_collection_analytics
        mock_get.return_value = mock_response
        
        analytics = analytics_client.get_collection_analytics("test_collection")
        
        assert isinstance(analytics, CollectionAnalytics)
        assert analytics.collection_name == "test_collection"
        assert analytics.total_points == 1000
        assert analytics.search_requests == 500
        assert analytics.cache_hit_rate == 0.92
        
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "analytics/collections/test_collection" in args[0]
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_get_usage_stats_success(self, mock_get, analytics_client, sample_usage_stats):
        """Test successful usage statistics retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_usage_stats
        mock_get.return_value = mock_response
        
        stats = analytics_client.get_usage_stats()
        
        assert isinstance(stats, UsageStats)
        assert stats.current_collections == 5
        assert stats.max_collections == 10
        assert stats.plan_name == "Professional"
        assert stats.collections_usage_percent == 50.0
        assert stats.points_usage_percent == 50.0
        
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "analytics/usage" in args[0]
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_get_region_performance_success(self, mock_get, analytics_client):
        """Test successful region performance retrieval."""
        region_data = {
            "us-east-1": {"latency_ms": 20.1, "requests_per_second": 60.0},
            "eu-central-1": {"latency_ms": 25.3, "requests_per_second": 45.0}
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = region_data
        mock_get.return_value = mock_response
        
        regions = analytics_client.get_region_performance("1h")
        
        assert isinstance(regions, dict)
        assert "us-east-1" in regions
        assert regions["us-east-1"]["latency_ms"] == 20.1
        
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["time_range"] == "1h"
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_get_cache_analytics_success(self, mock_get, analytics_client):
        """Test successful cache analytics retrieval."""
        cache_data = {
            "hit_rate": 0.89,
            "miss_rate": 0.11,
            "total_requests": 10000,
            "cache_size_mb": 512.5
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = cache_data
        mock_get.return_value = mock_response
        
        cache_stats = analytics_client.get_cache_analytics()
        
        assert cache_stats["hit_rate"] == 0.89
        assert cache_stats["total_requests"] == 10000
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_get_top_collections_success(self, mock_get, analytics_client):
        """Test successful top collections retrieval."""
        top_collections_data = [
            {"name": "collection1", "requests": 1000, "latency_ms": 15.2},
            {"name": "collection2", "requests": 800, "latency_ms": 18.7}
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = top_collections_data
        mock_get.return_value = mock_response
        
        top_collections = analytics_client.get_top_collections(
            metric="requests", time_range="7d", limit=5
        )
        
        assert len(top_collections) == 2
        assert top_collections[0]["name"] == "collection1"
        
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["metric"] == "requests"
        assert kwargs["params"]["limit"] == "5"


class TestAnalyticsErrorHandling:
    """Test error handling in analytics operations."""
    
    @pytest.fixture
    def analytics_client(self):
        """Analytics client fixture."""
        auth_headers = {"Authorization": "Bearer test_key"}
        return AnalyticsClient("https://test-api.aetherfy.com", auth_headers)
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_performance_analytics_error_handling(self, mock_get, analytics_client):
        """Test error handling in performance analytics retrieval."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "message": "Internal server error",
            "request_id": "req_123"
        }
        mock_response.content = True
        mock_get.return_value = mock_response
        
        with pytest.raises(AetherfyVectorsException):
            analytics_client.get_performance_analytics()
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_collection_analytics_not_found(self, mock_get, analytics_client):
        """Test collection analytics for non-existent collection."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "message": "Collection not found",
            "error_code": "collection_not_found",
            "request_id": "req_456"
        }
        mock_response.content = True
        mock_get.return_value = mock_response
        
        with pytest.raises(AetherfyVectorsException):
            analytics_client.get_collection_analytics("nonexistent_collection")
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_request_exception_handling(self, mock_get, analytics_client):
        """Test handling of request exceptions."""
        mock_get.side_effect = requests.RequestException("Network error")
        
        with pytest.raises(AetherfyVectorsException) as exc_info:
            analytics_client.get_usage_stats()
        
        assert "Failed to retrieve usage statistics" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)
    
    @patch('aetherfy_vectors.analytics.requests.get')
    def test_timeout_handling(self, mock_get, analytics_client):
        """Test timeout handling in analytics requests."""
        mock_get.side_effect = requests.Timeout("Request timed out")
        
        with pytest.raises(AetherfyVectorsException):
            analytics_client.get_performance_analytics()


class TestAnalyticsModels:
    """Test analytics data model functionality."""
    
    def test_performance_analytics_from_dict(self, sample_performance_analytics):
        """Test PerformanceAnalytics creation from dictionary."""
        analytics = PerformanceAnalytics.from_dict(sample_performance_analytics)
        
        assert analytics.cache_hit_rate == 0.85
        assert analytics.avg_latency_ms == 23.5
        assert analytics.requests_per_second == 150.0
        assert len(analytics.active_regions) == 3
        assert "us-east-1" in analytics.region_performance
        assert analytics.total_requests == 129600
        assert analytics.error_rate == 0.002
    
    def test_collection_analytics_from_dict(self, sample_collection_analytics):
        """Test CollectionAnalytics creation from dictionary."""
        analytics = CollectionAnalytics.from_dict(sample_collection_analytics)
        
        assert analytics.collection_name == "test_collection"
        assert analytics.total_points == 1000
        assert analytics.search_requests == 500
        assert analytics.avg_search_latency_ms == 18.5
        assert analytics.cache_hit_rate == 0.92
        assert analytics.storage_size_mb == 45.2
    
    def test_usage_stats_from_dict(self, sample_usage_stats):
        """Test UsageStats creation from dictionary."""
        stats = UsageStats.from_dict(sample_usage_stats)
        
        assert stats.current_collections == 5
        assert stats.max_collections == 10
        assert stats.plan_name == "Professional"
        
        # Test calculated properties
        assert stats.collections_usage_percent == 50.0
        assert stats.points_usage_percent == 50.0
        assert stats.requests_usage_percent == 25.0
        assert stats.storage_usage_percent == 25.05
    
    def test_usage_stats_percentage_calculations(self):
        """Test usage percentage calculations."""
        data = {
            "current_collections": 3,
            "max_collections": 10,
            "current_points": 75000,
            "max_points": 100000,
            "requests_this_month": 30000,
            "max_requests_per_month": 50000,
            "storage_used_mb": 400.0,
            "max_storage_mb": 800.0,
            "plan_name": "Starter"
        }
        
        stats = UsageStats.from_dict(data)
        
        assert stats.collections_usage_percent == 30.0
        assert stats.points_usage_percent == 75.0
        assert stats.requests_usage_percent == 60.0
        assert stats.storage_usage_percent == 50.0


class TestAnalyticsIntegration:
    """Test analytics integration with main client."""
    
    def test_client_analytics_integration(self, client, mock_analytics_client):
        """Test analytics integration in main client."""
        # Replace analytics client with mock
        client.analytics = mock_analytics_client(
            client.endpoint, client.auth_headers, client.timeout
        )
        
        # Test performance analytics
        perf_analytics = client.get_performance_analytics()
        assert isinstance(perf_analytics, PerformanceAnalytics)
        assert perf_analytics.cache_hit_rate == 0.85
        
        # Test collection analytics
        coll_analytics = client.get_collection_analytics("test_collection")
        assert isinstance(coll_analytics, CollectionAnalytics)
        assert coll_analytics.collection_name == "test_collection"
        
        # Test usage stats
        usage_stats = client.get_usage_stats()
        assert isinstance(usage_stats, UsageStats)
        assert usage_stats.current_collections == 5