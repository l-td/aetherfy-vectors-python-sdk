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

    @patch('time.sleep')
    def test_exponential_backoff(self, mock_sleep):
        """Should apply exponential backoff delays"""
        mock_fn = Mock(side_effect=ServiceUnavailableError("Service Unavailable"))

        with pytest.raises(ServiceUnavailableError):
            retry_with_backoff(mock_fn, max_retries=3, base_delay=0.1)

        # Should have 3 sleep calls (one per retry)
        assert mock_sleep.call_count == 3

        # Get the actual delay values
        delays = [call[0][0] for call in mock_sleep.call_args_list]

        # Base delays before jitter: 0.1, 0.2, 0.4 (exponential: base * 2^attempt)
        # With jitter (50-100%), delays should be in ranges:
        # delay1: 0.05-0.1, delay2: 0.1-0.2, delay3: 0.2-0.4
        assert 0.05 <= delays[0] <= 0.1
        assert 0.1 <= delays[1] <= 0.2
        assert 0.2 <= delays[2] <= 0.4

        # Verify exponential growth: third delay should be greater than first
        assert delays[2] > delays[0]

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

    @patch('time.sleep')
    def test_jitter_applied(self, mock_sleep):
        """Should apply jitter to prevent thundering herd"""
        mock_fn = Mock(side_effect=ServiceUnavailableError("Service Unavailable"))

        with pytest.raises(ServiceUnavailableError):
            retry_with_backoff(mock_fn, max_retries=2, base_delay=0.1)

        # Should have 2 sleep calls (one per retry)
        assert mock_sleep.call_count == 2

        # Get the actual delay values
        delays = [call[0][0] for call in mock_sleep.call_args_list]

        # Base delay is 0.1, with jitter (50-100%) should be between 0.05 and 0.1
        # Base delay for retry 2 is 0.2, with jitter should be between 0.1 and 0.2
        assert 0.05 <= delays[0] <= 0.1
        assert 0.1 <= delays[1] <= 0.2

        # Verify that jitter was actually applied (delays should not be exact exponential values)
        # With jitter, it's very unlikely (probability < 0.01) that both delays are at exact maximum
        assert not (delays[0] == 0.1 and delays[1] == 0.2)

    @patch('time.sleep')
    def test_max_delay_cap(self, mock_sleep):
        """Should cap delay at max_delay"""
        mock_fn = Mock(side_effect=ServiceUnavailableError("Service Unavailable"))

        with pytest.raises(ServiceUnavailableError):
            retry_with_backoff(
                mock_fn, max_retries=10, base_delay=1.0, max_delay=0.5
            )

        # Should have 10 sleep calls (one per retry)
        assert mock_sleep.call_count == 10

        # Get the actual delay values
        delays = [call[0][0] for call in mock_sleep.call_args_list]

        # Without cap: 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, ...
        # With max_delay=0.5, all delays after first should be capped at 0.5
        # With jitter (50-100%), delays should be in range [0.25, 0.5]
        for delay in delays:
            assert 0.25 <= delay <= 0.5

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
