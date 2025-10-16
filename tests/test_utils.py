"""
Tests for utility functions.

Tests validation, error parsing, and helper functions.
"""

import pytest
from aetherfy_vectors.utils import (
    validate_vector,
    validate_collection_name,
    validate_point_id,
    build_api_url,
    parse_error_response,
    format_points_for_upsert,
    sanitize_for_logging,
)
from aetherfy_vectors.exceptions import (
    ValidationError,
    AuthenticationError,
    RateLimitExceededError,
    ServiceUnavailableError,
    CollectionNotFoundError,
    PointNotFoundError,
    RequestTimeoutError,
    AetherfyVectorsException,
)


class TestValidateVector:
    """Test vector validation."""

    def test_validate_vector_success(self):
        """Test successful vector validation."""
        vector = [1.0, 2.0, 3.0]
        validate_vector(vector)  # Should not raise

    def test_validate_vector_with_dimension(self):
        """Test vector validation with expected dimension."""
        vector = [1.0, 2.0, 3.0]
        validate_vector(vector, expected_dimension=3)  # Should not raise

    def test_validate_vector_not_list(self):
        """Test vector validation fails for non-list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_vector("not a list")
        assert "must be a list of floats" in str(exc_info.value)

    def test_validate_vector_empty(self):
        """Test vector validation fails for empty vector."""
        with pytest.raises(ValidationError) as exc_info:
            validate_vector([])
        assert "cannot be empty" in str(exc_info.value)

    def test_validate_vector_non_numeric(self):
        """Test vector validation fails for non-numeric values."""
        with pytest.raises(ValidationError) as exc_info:
            validate_vector([1.0, "two", 3.0])
        assert "must contain only numeric values" in str(exc_info.value)

    def test_validate_vector_dimension_mismatch(self):
        """Test vector validation fails for dimension mismatch."""
        vector = [1.0, 2.0, 3.0]
        with pytest.raises(ValidationError) as exc_info:
            validate_vector(vector, expected_dimension=5)
        assert "dimension mismatch" in str(exc_info.value)
        assert "expected 5, got 3" in str(exc_info.value)

    def test_validate_vector_with_integers(self):
        """Test vector validation accepts integers."""
        vector = [1, 2, 3]
        validate_vector(vector)  # Should not raise


class TestValidateCollectionName:
    """Test collection name validation."""

    def test_validate_collection_name_success(self):
        """Test successful collection name validation."""
        validate_collection_name("my_collection")  # Should not raise

    def test_validate_collection_name_not_string(self):
        """Test collection name validation fails for non-string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_collection_name(123)
        assert "must be a string" in str(exc_info.value)

    def test_validate_collection_name_empty(self):
        """Test collection name validation fails for empty string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_collection_name("   ")
        assert "cannot be empty" in str(exc_info.value)

    def test_validate_collection_name_too_long(self):
        """Test collection name validation fails for too long name."""
        long_name = "a" * 256
        with pytest.raises(ValidationError) as exc_info:
            validate_collection_name(long_name)
        assert "255 characters or less" in str(exc_info.value)

    def test_validate_collection_name_invalid_characters(self):
        """Test collection name validation fails for invalid characters."""
        invalid_names = [
            "collection/name",
            "collection\\name",
            "collection?name",
            "collection%name",
            "collection*name",
            "collection:name",
            "collection|name",
            'collection"name',
            "collection<name",
            "collection>name",
        ]
        for name in invalid_names:
            with pytest.raises(ValidationError) as exc_info:
                validate_collection_name(name)
            assert "invalid characters" in str(exc_info.value)


class TestValidatePointId:
    """Test point ID validation."""

    def test_validate_point_id_string_success(self):
        """Test successful point ID validation with string."""
        validate_point_id("point_123")  # Should not raise

    def test_validate_point_id_int_success(self):
        """Test successful point ID validation with integer."""
        validate_point_id(123)  # Should not raise

    def test_validate_point_id_invalid_type(self):
        """Test point ID validation fails for invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_point_id([1, 2, 3])
        assert "must be a string or integer" in str(exc_info.value)

    def test_validate_point_id_empty_string(self):
        """Test point ID validation fails for empty string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_point_id("   ")
        assert "cannot be empty string" in str(exc_info.value)


class TestBuildApiUrl:
    """Test API URL building."""

    def test_build_api_url_basic(self):
        """Test basic API URL building."""
        url = build_api_url("https://api.example.com", "collections")
        assert url == "https://api.example.com/collections"

    def test_build_api_url_with_trailing_slash(self):
        """Test API URL building with trailing slash in base."""
        url = build_api_url("https://api.example.com/", "collections")
        assert url == "https://api.example.com/collections"

    def test_build_api_url_with_leading_slash(self):
        """Test API URL building with leading slash in endpoint."""
        url = build_api_url("https://api.example.com", "/collections")
        assert url == "https://api.example.com/collections"

    def test_build_api_url_with_both_slashes(self):
        """Test API URL building with both slashes."""
        url = build_api_url("https://api.example.com/", "/collections")
        assert url == "https://api.example.com/collections"


class TestParseErrorResponse:
    """Test error response parsing."""

    def test_parse_error_response_401(self):
        """Test parsing 401 authentication error."""
        response_data = {
            "message": "Unauthorized",
            "request_id": "req_123",
            "error_code": "auth_error",
        }
        error = parse_error_response(response_data, 401)
        assert isinstance(error, AuthenticationError)
        assert "Unauthorized" in str(error)

    def test_parse_error_response_429(self):
        """Test parsing 429 rate limit error."""
        response_data = {
            "message": "Rate limit exceeded",
            "request_id": "req_123",
            "details": {"retry_after": 60},
        }
        error = parse_error_response(response_data, 429)
        assert isinstance(error, RateLimitExceededError)
        assert error.retry_after == 60

    def test_parse_error_response_502(self):
        """Test parsing 502 service unavailable error."""
        response_data = {"message": "Bad Gateway", "request_id": "req_123"}
        error = parse_error_response(response_data, 502)
        assert isinstance(error, ServiceUnavailableError)

    def test_parse_error_response_503(self):
        """Test parsing 503 service unavailable error."""
        response_data = {"message": "Service Unavailable", "request_id": "req_123"}
        error = parse_error_response(response_data, 503)
        assert isinstance(error, ServiceUnavailableError)

    def test_parse_error_response_504(self):
        """Test parsing 504 gateway timeout error."""
        response_data = {"message": "Gateway Timeout", "request_id": "req_123"}
        error = parse_error_response(response_data, 504)
        assert isinstance(error, ServiceUnavailableError)

    def test_parse_error_response_404_collection(self):
        """Test parsing 404 collection not found error."""
        response_data = {
            "message": "Collection not found",
            "request_id": "req_123",
            "error_code": "collection_not_found",
            "details": {"collection_name": "test_collection"},
        }
        error = parse_error_response(response_data, 404)
        assert isinstance(error, CollectionNotFoundError)
        assert "test_collection" in str(error)

    def test_parse_error_response_404_point(self):
        """Test parsing 404 point not found error."""
        response_data = {
            "message": "Point not found",
            "request_id": "req_123",
            "error_code": "point_not_found",
            "details": {"point_id": "point_123", "collection_name": "test_collection"},
        }
        error = parse_error_response(response_data, 404)
        assert isinstance(error, PointNotFoundError)

    def test_parse_error_response_400(self):
        """Test parsing 400 validation error."""
        response_data = {"message": "Bad Request", "request_id": "req_123"}
        error = parse_error_response(response_data, 400)
        assert isinstance(error, ValidationError)

    def test_parse_error_response_408(self):
        """Test parsing 408 request timeout error."""
        response_data = {"message": "Request Timeout", "request_id": "req_123"}
        error = parse_error_response(response_data, 408)
        assert isinstance(error, RequestTimeoutError)

    def test_parse_error_response_generic(self):
        """Test parsing generic error."""
        response_data = {"message": "Unknown error", "request_id": "req_123"}
        error = parse_error_response(response_data, 500)
        assert isinstance(error, AetherfyVectorsException)
        assert not isinstance(error, ValidationError)


class TestFormatPointsForUpsert:
    """Test points formatting for upsert."""

    def test_format_points_for_upsert_success(self):
        """Test successful points formatting."""
        points = [
            {"id": "point_1", "vector": [1.0, 2.0, 3.0], "payload": {"key": "value"}},
            {"id": "point_2", "vector": [4.0, 5.0, 6.0]},
        ]
        formatted = format_points_for_upsert(points)
        assert len(formatted) == 2
        assert formatted[0]["id"] == "point_1"
        assert formatted[0]["vector"] == [1.0, 2.0, 3.0]
        assert formatted[0]["payload"] == {"key": "value"}
        assert formatted[1]["id"] == "point_2"
        assert "payload" not in formatted[1]

    def test_format_points_for_upsert_not_list(self):
        """Test points formatting fails for non-list."""
        with pytest.raises(ValidationError) as exc_info:
            format_points_for_upsert("not a list")
        assert "must be a list" in str(exc_info.value)

    def test_format_points_for_upsert_empty_list(self):
        """Test points formatting fails for empty list."""
        with pytest.raises(ValidationError) as exc_info:
            format_points_for_upsert([])
        assert "cannot be empty" in str(exc_info.value)

    def test_format_points_for_upsert_point_not_dict(self):
        """Test points formatting fails when point is not dict."""
        points = [{"id": "point_1", "vector": [1.0]}, "not a dict"]
        with pytest.raises(ValidationError) as exc_info:
            format_points_for_upsert(points)
        assert "must be a dictionary" in str(exc_info.value)
        assert "index 1" in str(exc_info.value)

    def test_format_points_for_upsert_missing_id(self):
        """Test points formatting fails when id is missing."""
        points = [{"vector": [1.0, 2.0, 3.0]}]
        with pytest.raises(ValidationError) as exc_info:
            format_points_for_upsert(points)
        assert "must have an 'id' field" in str(exc_info.value)
        assert "index 0" in str(exc_info.value)

    def test_format_points_for_upsert_missing_vector(self):
        """Test points formatting fails when vector is missing."""
        points = [{"id": "point_1"}]
        with pytest.raises(ValidationError) as exc_info:
            format_points_for_upsert(points)
        assert "must have a 'vector' field" in str(exc_info.value)
        assert "index 0" in str(exc_info.value)


class TestSanitizeForLogging:
    """Test sanitization for logging."""

    def test_sanitize_simple_dict(self):
        """Test sanitizing simple dictionary."""
        data = {"name": "test", "value": 123}
        sanitized = sanitize_for_logging(data)
        assert sanitized == {"name": "test", "value": 123}

    def test_sanitize_dict_with_sensitive_keys(self):
        """Test sanitizing dictionary with sensitive keys."""
        data = {
            "api_key": "secret123",
            "token": "token123",
            "password": "pass123",
            "secret": "secret123",
            "name": "test",
        }
        sanitized = sanitize_for_logging(data)
        assert sanitized["api_key"] == "***"
        assert sanitized["token"] == "***"
        assert sanitized["password"] == "***"
        assert sanitized["secret"] == "***"
        assert sanitized["name"] == "test"

    def test_sanitize_nested_dict(self):
        """Test sanitizing nested dictionary."""
        data = {
            "user": {"name": "test", "api_key": "secret123"},
            "config": {"timeout": 30},
        }
        sanitized = sanitize_for_logging(data)
        assert sanitized["user"]["name"] == "test"
        assert sanitized["user"]["api_key"] == "***"
        assert sanitized["config"]["timeout"] == 30

    def test_sanitize_list(self):
        """Test sanitizing list."""
        data = [
            {"name": "test1", "password": "pass1"},
            {"name": "test2", "password": "pass2"},
        ]
        sanitized = sanitize_for_logging(data)
        assert sanitized[0]["name"] == "test1"
        assert sanitized[0]["password"] == "***"
        assert sanitized[1]["name"] == "test2"
        assert sanitized[1]["password"] == "***"

    def test_sanitize_long_string(self):
        """Test sanitizing very long string."""
        long_string = "a" * 150
        sanitized = sanitize_for_logging(long_string)
        assert len(sanitized) == 103  # 100 chars + "..."
        assert sanitized.endswith("...")

    def test_sanitize_primitive_types(self):
        """Test sanitizing primitive types."""
        assert sanitize_for_logging(123) == 123
        assert sanitize_for_logging("short") == "short"
        assert sanitize_for_logging(True) is True
        assert sanitize_for_logging(None) is None
