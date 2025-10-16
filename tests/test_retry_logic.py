"""
Tests for retry logic with HTTP errors.
"""

import pytest
import time
from unittest.mock import Mock, patch
from aetherfy_vectors.utils import retry_with_backoff
from aetherfy_vectors.exceptions import (
    ServiceUnavailableError,
    RequestTimeoutError,
    NetworkError,
    RateLimitExceededError,
    ValidationError,
    AuthenticationError,
    is_retryable_error,
)


class TestRetryLogic:
    def test_retry_on_503(self):
        """Should retry on 503 Service Unavailable"""
        mock_fn = Mock(
            side_effect=[ServiceUnavailableError("Service Unavailable"), "success"]
        )

        result = retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)

        assert result == "success"
        assert mock_fn.call_count == 2

    def test_retry_on_502(self):
        """Should retry on 502 Bad Gateway"""
        mock_fn = Mock(
            side_effect=[ServiceUnavailableError("Bad Gateway"), "success"]
        )

        result = retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert mock_fn.call_count == 2

    def test_retry_on_timeout(self):
        """Should retry on timeout errors"""
        mock_fn = Mock(side_effect=[RequestTimeoutError("Timeout"), "success"])

        result = retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert mock_fn.call_count == 2

    def test_retry_on_network_error(self):
        """Should retry on network errors"""
        mock_fn = Mock(side_effect=[NetworkError("Network error"), "success"])

        result = retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert mock_fn.call_count == 2

    def test_no_retry_on_validation_error(self):
        """Should NOT retry on 400 ValidationError"""
        mock_fn = Mock(side_effect=ValidationError("Bad request"))

        with pytest.raises(ValidationError):
            retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)

        assert mock_fn.call_count == 1

    def test_no_retry_on_auth_error(self):
        """Should NOT retry on 401 AuthenticationError"""
        mock_fn = Mock(side_effect=AuthenticationError("Unauthorized"))

        with pytest.raises(AuthenticationError):
            retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)

        assert mock_fn.call_count == 1

    def test_exponential_backoff(self):
        """Should apply exponential backoff delays"""
        timestamps = []

        def failing_fn():
            timestamps.append(time.time())
            raise ServiceUnavailableError("Service Unavailable")

        with pytest.raises(ServiceUnavailableError):
            retry_with_backoff(failing_fn, max_retries=3, base_delay=0.1)

        # Check that delays increase exponentially
        assert len(timestamps) == 4  # Initial + 3 retries
        delay1 = timestamps[1] - timestamps[0]
        delay2 = timestamps[2] - timestamps[1]
        assert delay2 > delay1

    def test_max_retries_exhausted(self):
        """Should fail after exhausting retries"""
        mock_fn = Mock(side_effect=ServiceUnavailableError("Service Unavailable"))

        with pytest.raises(ServiceUnavailableError):
            retry_with_backoff(mock_fn, max_retries=2, base_delay=0.01)

        assert mock_fn.call_count == 3  # Initial + 2 retries

    def test_retry_condition_custom(self):
        """Should respect custom retry condition"""

        def custom_condition(error):
            return "should_retry" in str(error)

        mock_fn = Mock(side_effect=ValueError("do not retry"))

        with pytest.raises(ValueError):
            retry_with_backoff(
                mock_fn,
                max_retries=3,
                base_delay=0.01,
                retry_condition=custom_condition,
            )

        assert mock_fn.call_count == 1

    def test_jitter_applied(self):
        """Should apply jitter to prevent thundering herd"""
        timestamps = []

        def failing_fn():
            timestamps.append(time.time())
            raise ServiceUnavailableError("Service Unavailable")

        with pytest.raises(ServiceUnavailableError):
            retry_with_backoff(failing_fn, max_retries=2, base_delay=0.1)

        # Check that delays have jitter (not exact exponential)
        assert len(timestamps) == 3  # Initial + 2 retries
        delay1 = timestamps[1] - timestamps[0]
        # Base delay is 0.1, with jitter should be between 0.05 and 0.1
        assert 0.05 <= delay1 <= 0.1

    def test_max_delay_cap(self):
        """Should cap delay at max_delay"""
        timestamps = []

        def failing_fn():
            timestamps.append(time.time())
            raise ServiceUnavailableError("Service Unavailable")

        with pytest.raises(ServiceUnavailableError):
            retry_with_backoff(
                failing_fn, max_retries=10, base_delay=1.0, max_delay=0.5
            )

        # Check that delays don't exceed max_delay
        for i in range(1, len(timestamps)):
            delay = timestamps[i] - timestamps[i - 1]
            # With jitter, delay should be between 50% and 100% of max_delay
            assert delay <= 0.5

    def test_successful_first_attempt(self):
        """Should return immediately on successful first attempt"""
        mock_fn = Mock(return_value="success")

        result = retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)

        assert result == "success"
        assert mock_fn.call_count == 1

    def test_retry_on_rate_limit_with_retry_after(self):
        """Should retry on rate limit error with retry_after"""
        mock_fn = Mock(
            side_effect=[
                RateLimitExceededError("Rate limit", retry_after=2),
                "success",
            ]
        )

        result = retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert mock_fn.call_count == 2

    def test_no_retry_on_rate_limit_without_retry_after(self):
        """Should NOT retry on rate limit error without retry_after"""
        mock_fn = Mock(side_effect=RateLimitExceededError("Rate limit"))

        with pytest.raises(RateLimitExceededError):
            retry_with_backoff(mock_fn, max_retries=3, base_delay=0.01)

        assert mock_fn.call_count == 1


class TestIsRetryableError:
    def test_service_unavailable_is_retryable(self):
        assert is_retryable_error(ServiceUnavailableError()) is True

    def test_timeout_is_retryable(self):
        assert is_retryable_error(RequestTimeoutError()) is True

    def test_network_error_is_retryable(self):
        assert is_retryable_error(NetworkError()) is True

    def test_rate_limit_with_retry_after_is_retryable(self):
        error = RateLimitExceededError("Rate limit", retry_after=2)
        assert is_retryable_error(error) is True

    def test_rate_limit_without_retry_after_is_not_retryable(self):
        error = RateLimitExceededError("Rate limit")
        assert is_retryable_error(error) is False

    def test_validation_error_is_not_retryable(self):
        assert is_retryable_error(ValidationError()) is False

    def test_auth_error_is_not_retryable(self):
        assert is_retryable_error(AuthenticationError()) is False

    def test_generic_exception_is_not_retryable(self):
        """Should not retry on generic exceptions"""
        assert is_retryable_error(ValueError("Generic error")) is False

    def test_rate_limit_with_none_retry_after_is_not_retryable(self):
        """Should not retry on rate limit with explicit None retry_after"""
        error = RateLimitExceededError("Rate limit", retry_after=None)
        assert is_retryable_error(error) is False
