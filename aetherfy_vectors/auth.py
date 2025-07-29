"""
Authentication and API key management for Aetherfy Vectors SDK.

Handles secure API key validation, header injection, and authentication
error handling for all requests to the global vector database service.
"""

import os
import re
from typing import Dict, Optional
from .exceptions import AuthenticationError


class APIKeyManager:
    """Manages API key authentication for Aetherfy Vectors."""

    API_KEY_PREFIX = "afy_"
    API_KEY_PATTERN = re.compile(r"^afy_(live|test)_[a-zA-Z0-9]{16,}$")

    def __init__(self, api_key: Optional[str] = None):
        """Initialize API key manager.

        Args:
            api_key: The API key. If None, will try to get from environment.

        Raises:
            AuthenticationError: If no valid API key is found.
        """
        self.api_key = self._resolve_api_key(api_key)
        self._validate_api_key(self.api_key)

    def _resolve_api_key(self, api_key: Optional[str]) -> str:
        """Resolve API key from parameter or environment.

        Args:
            api_key: Explicit API key or None.

        Returns:
            The resolved API key.

        Raises:
            AuthenticationError: If no API key is found.
        """
        if api_key:
            return api_key

        # Try environment variables
        env_key = os.getenv("AETHERFY_API_KEY") or os.getenv("AETHERFY_VECTORS_API_KEY")
        if env_key:
            return env_key

        raise AuthenticationError(
            "No API key provided. Set AETHERFY_API_KEY environment variable "
            "or pass api_key parameter to AetherfyVectorsClient."
        )

    def _validate_api_key(self, api_key: str) -> None:
        """Validate API key format.

        Args:
            api_key: The API key to validate.

        Raises:
            AuthenticationError: If API key format is invalid.
        """
        if not isinstance(api_key, str):
            raise AuthenticationError("API key must be a string")

        if not api_key.strip():
            raise AuthenticationError("API key cannot be empty")

        if not self.API_KEY_PATTERN.match(api_key):
            raise AuthenticationError(
                "Invalid API key format. API key should start with 'afy_live_' "
                "or 'afy_test_' followed by at least 16 alphanumeric characters."
            )

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests.

        Returns:
            Dictionary containing authentication headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
        }

    def is_test_key(self) -> bool:
        """Check if the API key is a test key.

        Returns:
            True if this is a test key, False if live key.
        """
        return self.api_key.startswith("afy_test_")

    def is_live_key(self) -> bool:
        """Check if the API key is a live key.

        Returns:
            True if this is a live key, False if test key.
        """
        return self.api_key.startswith("afy_live_")

    def mask_api_key(self) -> str:
        """Get a masked version of the API key for logging.

        Returns:
            Masked API key for secure logging.
        """
        return "***"

    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Static method to validate API key format without creating instance.

        Args:
            api_key: The API key to validate.

        Returns:
            True if format is valid, False otherwise.
        """
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        return bool(APIKeyManager.API_KEY_PATTERN.match(api_key))
