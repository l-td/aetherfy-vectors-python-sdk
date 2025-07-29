"""
Tests for authentication and API key management functionality.

Tests API key validation, authentication headers, and error handling.
"""

import pytest
import os
from unittest.mock import patch

from aetherfy_vectors.auth import APIKeyManager
from aetherfy_vectors.exceptions import AuthenticationError


class TestAPIKeyManager:
    """Test API key management functionality."""
    
    def test_valid_live_api_key(self):
        """Test initialization with valid live API key."""
        api_key = "afy_live_1234567890abcdef"
        manager = APIKeyManager(api_key)
        
        assert manager.api_key == api_key
        assert manager.is_live_key()
        assert not manager.is_test_key()
    
    def test_valid_test_api_key(self):
        """Test initialization with valid test API key."""
        api_key = "afy_test_9876543210fedcba"
        manager = APIKeyManager(api_key)
        
        assert manager.api_key == api_key
        assert manager.is_test_key()
        assert not manager.is_live_key()
    
    def test_api_key_from_environment_aetherfy_api_key(self):
        """Test API key resolution from AETHERFY_API_KEY environment variable."""
        api_key = "afy_live_1234567890abcdef123456"
        
        with patch.dict(os.environ, {"AETHERFY_API_KEY": api_key}):
            manager = APIKeyManager()
            assert manager.api_key == api_key
    
    def test_api_key_from_environment_aetherfy_vectors_api_key(self):
        """Test API key resolution from AETHERFY_VECTORS_API_KEY environment variable."""
        api_key = "afy_test_1234567890abcdef123456"
        
        with patch.dict(os.environ, {"AETHERFY_VECTORS_API_KEY": api_key}):
            manager = APIKeyManager()
            assert manager.api_key == api_key
    
    def test_explicit_api_key_overrides_environment(self):
        """Test that explicit API key overrides environment variables."""
        env_key = "afy_live_1234567890abcdef123456"
        explicit_key = "afy_test_1234567890abcdef123456"
        
        with patch.dict(os.environ, {"AETHERFY_API_KEY": env_key}):
            manager = APIKeyManager(explicit_key)
            assert manager.api_key == explicit_key
    
    def test_no_api_key_raises_error(self):
        """Test that missing API key raises AuthenticationError."""
        with pytest.raises(AuthenticationError) as exc_info:
            APIKeyManager()
        
        assert "No API key provided" in str(exc_info.value)
        assert "AETHERFY_API_KEY" in str(exc_info.value)
    
    def test_invalid_api_key_format_raises_error(self):
        """Test that invalid API key format raises AuthenticationError."""
        invalid_keys = [
            "invalid_key",
            "afy_invalid_key", 
            "afy_live_",
            "afy_test_short",
            "not_afy_live_1234567890abcdef",
            "afy_live_123!@#$%^&*()",
        ]
        
        for invalid_key in invalid_keys:
            with pytest.raises(AuthenticationError) as exc_info:
                APIKeyManager(invalid_key)
            
            assert "Invalid API key format" in str(exc_info.value)
        
        # Test empty string separately as it triggers different error path
        with pytest.raises(AuthenticationError) as exc_info:
            APIKeyManager("")
        assert "No API key provided" in str(exc_info.value)
    
    def test_empty_api_key_raises_error(self):
        """Test that empty API key raises AuthenticationError."""
        with pytest.raises(AuthenticationError):
            APIKeyManager("")
    
    def test_non_string_api_key_raises_error(self):
        """Test that non-string API key raises AuthenticationError."""
        with pytest.raises(AuthenticationError):
            APIKeyManager(12345)
    
    def test_whitespace_only_api_key_raises_error(self):
        """Test that whitespace-only API key raises AuthenticationError."""
        with pytest.raises(AuthenticationError):
            APIKeyManager("   ")


class TestAPIKeyValidation:
    """Test API key validation methods."""
    
    def test_validate_api_key_format_valid_keys(self):
        """Test validate_api_key_format with valid keys."""
        valid_keys = [
            "afy_live_1234567890abcdef",
            "afy_test_9876543210fedcba",
            "afy_live_abcdef1234567890ABCDEF",
            "afy_test_verylongkeywithalotofcharacters123456789",
        ]
        
        for key in valid_keys:
            assert APIKeyManager.validate_api_key_format(key) is True
    
    def test_validate_api_key_format_invalid_keys(self):
        """Test validate_api_key_format with invalid keys."""
        invalid_keys = [
            "invalid_key",
            "afy_invalid_key",
            "afy_live_",
            "afy_test_short",
            "",
            None,
            123,
            "afy_live_123!@#$%^",
        ]
        
        for key in invalid_keys:
            assert APIKeyManager.validate_api_key_format(key) is False


class TestAuthenticationHeaders:
    """Test authentication header generation."""
    
    def test_get_auth_headers(self):
        """Test authentication headers generation."""
        api_key = "afy_live_1234567890abcdef"
        manager = APIKeyManager(api_key)
        
        headers = manager.get_auth_headers()
        
        assert "Authorization" in headers
        assert "X-API-Key" in headers
        assert headers["Authorization"] == f"Bearer {api_key}"
        assert headers["X-API-Key"] == api_key
    
    def test_auth_headers_are_dict(self):
        """Test that auth headers return a dictionary."""
        api_key = "afy_test_1234567890abcdef"
        manager = APIKeyManager(api_key)
        
        headers = manager.get_auth_headers()
        
        assert isinstance(headers, dict)
        assert len(headers) == 2


class TestAPIKeyUtilities:
    """Test API key utility methods."""
    
    def test_mask_api_key(self):
        """Test API key masking for logging."""
        api_key = "afy_live_1234567890abcdef"
        manager = APIKeyManager(api_key)
        
        masked = manager.mask_api_key()
        
        assert masked == "***"
    
    def test_mask_short_api_key(self):
        """Test masking of very short API key."""
        # This shouldn't happen with valid keys, but test edge case
        manager = APIKeyManager("afy_live_1234567890abcdef")
        manager.api_key = "short"  # Manually set for testing
        
        masked = manager.mask_api_key()
        assert masked == "***"
    
    def test_is_live_key(self):
        """Test live key detection."""
        live_manager = APIKeyManager("afy_live_1234567890abcdef")
        test_manager = APIKeyManager("afy_test_1234567890abcdef")
        
        assert live_manager.is_live_key() is True
        assert test_manager.is_live_key() is False
    
    def test_is_test_key(self):
        """Test test key detection."""
        live_manager = APIKeyManager("afy_live_1234567890abcdef")
        test_manager = APIKeyManager("afy_test_1234567890abcdef")
        
        assert live_manager.is_test_key() is False
        assert test_manager.is_test_key() is True


class TestAPIKeyManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_api_key_pattern_matching(self):
        """Test API key pattern matching edge cases."""
        # Valid edge cases
        valid_edge_cases = [
            "afy_live_" + "a" * 16,  # Minimum length
            "afy_test_" + "1" * 50,  # Long key
            "afy_live_ABC123def456GHIJ",  # Mixed case and numbers
        ]
        
        for key in valid_edge_cases:
            manager = APIKeyManager(key)
            assert manager.api_key == key
    
    def test_api_key_prefix_validation(self):
        """Test that only correct prefixes are accepted."""
        invalid_prefixes = [
            "afx_live_1234567890abcdef",  # Wrong prefix
            "afy_prod_1234567890abcdef",  # Invalid environment
            "afy_dev_1234567890abcdef",   # Invalid environment
        ]
        
        for key in invalid_prefixes:
            with pytest.raises(AuthenticationError):
                APIKeyManager(key)
    
    def test_api_key_case_sensitivity(self):
        """Test that API key validation is case sensitive for prefix."""
        invalid_case_keys = [
            "AFY_LIVE_1234567890abcdef",  # Uppercase prefix
            "Afy_Live_1234567890abcdef",  # Mixed case prefix
            "afy_LIVE_1234567890abcdef",  # Uppercase environment
        ]
        
        for key in invalid_case_keys:
            with pytest.raises(AuthenticationError):
                APIKeyManager(key)