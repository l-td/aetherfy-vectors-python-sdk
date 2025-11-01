"""
Analytics retrieval functionality for Aetherfy Vectors SDK.

Provides methods to retrieve performance analytics, usage statistics,
and insights from the global vector database service.
"""

from typing import Dict, Any, Optional, List
import requests

from .models import PerformanceAnalytics, CollectionAnalytics, UsageStats
from .exceptions import AetherfyVectorsException
from .utils import parse_error_response, build_api_url


class AnalyticsClient:
    """Client for retrieving analytics data from Aetherfy backend."""

    def __init__(
        self,
        base_url: str,
        auth_headers: Dict[str, str],
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ):
        """Initialize analytics client.

        Args:
            base_url: Base URL for API requests.
            auth_headers: Authentication headers.
            timeout: Request timeout in seconds.
            session: Optional requests Session for connection pooling.
        """
        self.base_url = base_url
        self.auth_headers = auth_headers
        self.timeout = timeout
        self.session = session if session is not None else requests

    def get_performance_analytics(
        self, time_range: str = "24h", region: Optional[str] = None
    ) -> PerformanceAnalytics:
        """Retrieve global performance analytics.

        Args:
            time_range: Time range for analytics (1h, 24h, 7d, 30d).
            region: Specific region to filter by (optional).

        Returns:
            Performance analytics data.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        params = {"time_range": time_range}
        if region:
            params["region"] = region

        url = build_api_url(self.base_url, "analytics/performance")

        try:
            response = self.session.get(
                url, headers=self.auth_headers, params=params, timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return PerformanceAnalytics.from_dict(data)
            else:
                error_data = response.json() if response.content else {}
                raise parse_error_response(error_data, response.status_code)

        except requests.RequestException as e:
            raise AetherfyVectorsException(
                f"Failed to retrieve performance analytics: {str(e)}"
            )

    def get_collection_analytics(
        self, collection_name: str, time_range: str = "24h"
    ) -> CollectionAnalytics:
        """Retrieve analytics for a specific collection.

        Args:
            collection_name: Name of the collection.
            time_range: Time range for analytics (1h, 24h, 7d, 30d).

        Returns:
            Collection-specific analytics data.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        params = {"time_range": time_range}
        url = build_api_url(self.base_url, f"analytics/collections/{collection_name}")

        try:
            response = self.session.get(
                url, headers=self.auth_headers, params=params, timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return CollectionAnalytics.from_dict(data)
            else:
                error_data = response.json() if response.content else {}
                raise parse_error_response(error_data, response.status_code)

        except requests.RequestException as e:
            raise AetherfyVectorsException(
                f"Failed to retrieve collection analytics: {str(e)}"
            )

    def get_usage_stats(self) -> UsageStats:
        """Retrieve current usage statistics against customer limits.

        Returns:
            Current usage statistics.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        url = build_api_url(self.base_url, "analytics/usage")

        try:
            response = self.session.get(
                url, headers=self.auth_headers, timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return UsageStats.from_dict(data)
            else:
                error_data = response.json() if response.content else {}
                raise parse_error_response(error_data, response.status_code)

        except requests.RequestException as e:
            raise AetherfyVectorsException(
                f"Failed to retrieve usage statistics: {str(e)}"
            )

    def get_region_performance(
        self, time_range: str = "24h"
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve performance metrics by region.

        Args:
            time_range: Time range for analytics (1h, 24h, 7d, 30d).

        Returns:
            Dictionary mapping region names to performance metrics.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        params = {"time_range": time_range}
        url = build_api_url(self.base_url, "analytics/regions")

        try:
            response = self.session.get(
                url, headers=self.auth_headers, params=params, timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.content else {}
                raise parse_error_response(error_data, response.status_code)

        except requests.RequestException as e:
            raise AetherfyVectorsException(
                f"Failed to retrieve region performance: {str(e)}"
            )

    def get_cache_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Retrieve cache performance analytics.

        Args:
            time_range: Time range for analytics (1h, 24h, 7d, 30d).

        Returns:
            Cache performance data.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        params = {"time_range": time_range}
        url = build_api_url(self.base_url, "analytics/cache")

        try:
            response = self.session.get(
                url, headers=self.auth_headers, params=params, timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.content else {}
                raise parse_error_response(error_data, response.status_code)

        except requests.RequestException as e:
            raise AetherfyVectorsException(
                f"Failed to retrieve cache analytics: {str(e)}"
            )

    def get_top_collections(
        self, metric: str = "requests", time_range: str = "24h", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve top collections by specified metric.

        Args:
            metric: Metric to sort by (requests, latency, storage).
            time_range: Time range for analytics (1h, 24h, 7d, 30d).
            limit: Number of collections to return.

        Returns:
            List of top collections with metrics.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        params = {
            "metric": str(metric),
            "time_range": str(time_range),
            "limit": str(limit),
        }
        url = build_api_url(self.base_url, "analytics/collections/top")

        try:
            response = self.session.get(
                url, headers=self.auth_headers, params=params, timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.content else {}
                raise parse_error_response(error_data, response.status_code)

        except requests.RequestException as e:
            raise AetherfyVectorsException(
                f"Failed to retrieve top collections: {str(e)}"
            )
