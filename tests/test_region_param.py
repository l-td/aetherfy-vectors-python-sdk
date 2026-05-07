"""
Tests for the WS9 region= constructor parameter and /api/v1/regions
discovery client on AetherfyVectorsClient.

Pins the contract:
  - region= validates against the Fly set (iad/fra/sin) at construction.
  - region= triggers a single GET /api/v1/regions, cached per instance.
  - AETHERFY_VECTORS_URL takes precedence over region= (logs a warning).
  - Discovery failure raises AetherfyVectorsException with a clear message.
"""

import pytest
from unittest.mock import Mock, patch

from aetherfy_vectors import AetherfyVectorsClient, CollectionInOtherRegionError
from aetherfy_vectors.exceptions import AetherfyVectorsException
from aetherfy_vectors.utils import parse_error_response


def _mock_regions_response(payload, status=200):
    resp = Mock()
    resp.status_code = status
    import json
    resp.content = json.dumps(payload).encode("utf-8")
    return resp


class TestRegionParamValidation:
    def test_invalid_region_raises_at_construction(self, api_key, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with pytest.raises(ValueError, match="region must be one of"):
            AetherfyVectorsClient(api_key=api_key, region="us-east-1")

    def test_valid_region_via_env_var(self, api_key, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        monkeypatch.setenv("AETHERFY_VECTORS_REGION", "fra")
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            mock_get.return_value = _mock_regions_response({
                "iad": "https://vectors-iad.aetherfy.run",
                "fra": "https://vectors-fra.aetherfy.run",
            })
            client = AetherfyVectorsClient(api_key=api_key)
            assert client.region == "fra"
            assert client.endpoint == "https://vectors-fra.aetherfy.run"


class TestRegionDiscovery:
    def test_region_resolves_via_discovery(self, api_key, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            mock_get.return_value = _mock_regions_response({
                "iad": "https://vectors-iad.aetherfy.run",
                "fra": "https://vectors-fra.aetherfy.run",
                "sin": "https://vectors-sin.aetherfy.run",
            })
            client = AetherfyVectorsClient(api_key=api_key, region="fra")
            assert client.endpoint == "https://vectors-fra.aetherfy.run"
            assert mock_get.call_count == 1
            # Verify it hit the default global URL's /api/v1/regions.
            url_arg = mock_get.call_args.args[0]
            assert url_arg.endswith("/api/v1/regions")

    def test_discovery_cached_across_operations(self, api_key, monkeypatch):
        # The cache lives on the client instance, but constructing one
        # client only triggers one discovery call. Construct two clients
        # and verify each does its own discovery (not module-shared).
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            mock_get.return_value = _mock_regions_response({
                "iad": "https://vectors-iad.aetherfy.run",
                "fra": "https://vectors-fra.aetherfy.run",
            })
            c1 = AetherfyVectorsClient(api_key=api_key, region="iad")
            c2 = AetherfyVectorsClient(api_key=api_key, region="fra")
            # Each instance fetched independently — no module-global cache.
            assert mock_get.call_count == 2
            assert c1.endpoint != c2.endpoint

    def test_discovery_failure_raises_clear_error(self, api_key, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            mock_get.return_value = _mock_regions_response({}, status=500)
            with pytest.raises(AetherfyVectorsException, match="discovery returned 500"):
                AetherfyVectorsClient(api_key=api_key, region="fra")

    def test_discovery_missing_region_raises(self, api_key, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            mock_get.return_value = _mock_regions_response({
                "iad": "https://vectors-iad.aetherfy.run",
            })
            with pytest.raises(AetherfyVectorsException, match="not configured at the discovery endpoint"):
                AetherfyVectorsClient(api_key=api_key, region="fra")


class TestEnvVarPrecedence:
    def test_env_var_wins_over_region_param(self, api_key, monkeypatch, caplog):
        # AETHERFY_VECTORS_URL is the production-agent injection from the
        # control-plane. region= is a local-dev knob. Env var must win
        # and a warning must be logged so a developer accidentally
        # hitting this in prod sees it.
        monkeypatch.setenv("AETHERFY_VECTORS_URL", "http://10.0.10.243:3000")
        import logging
        caplog.set_level(logging.WARNING, logger="aetherfy_vectors.client")
        client = AetherfyVectorsClient(api_key=api_key, region="fra")
        assert client.endpoint == "http://10.0.10.243:3000"
        assert any(
            "AETHERFY_VECTORS_URL" in rec.getMessage() and "region=" in rec.getMessage()
            for rec in caplog.records
        )

    def test_explicit_endpoint_skips_discovery(self, api_key, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            client = AetherfyVectorsClient(
                api_key=api_key,
                endpoint="http://localhost:3000",
                region="fra",
            )
            # Explicit endpoint wins; no discovery call.
            assert client.endpoint == "http://localhost:3000"
            assert mock_get.call_count == 0


class TestCollectionInOtherRegionError:
    """parse_error_response → CollectionInOtherRegionError on the 409.

    Pins the typed-exception contract so callers can branch on
    isinstance(...) and read .existing_regions / .requesting_region
    without parsing message strings.
    """

    def test_typed_error_carries_fields(self):
        body = {
            "error": {
                "code": "COLLECTION_EXISTS_IN_OTHER_REGION",
                "message": "Collection 'foo' already exists in region fra. ...",
                "collection_name": "foo",
                "existing_regions": ["fra"],
                "requesting_region": "iad",
            }
        }
        err = parse_error_response(body, 409)
        assert isinstance(err, CollectionInOtherRegionError)
        assert err.collection_name == "foo"
        assert err.existing_regions == ["fra"]
        assert err.requesting_region == "iad"
        assert err.status_code == 409
        # The base exception preserves error_code so generic handlers
        # can still string-match if they want.
        assert err.error_code == "COLLECTION_EXISTS_IN_OTHER_REGION"

    def test_other_409s_still_become_generic(self):
        # A 409 without a recognized error_code falls through to the
        # generic AetherfyVectorsException — matches existing behavior
        # and avoids over-typing every conflict case.
        body = {"error": {"code": "SOMETHING_ELSE", "message": "nope"}}
        err = parse_error_response(body, 409)
        assert not isinstance(err, CollectionInOtherRegionError)
        assert err.status_code == 409
