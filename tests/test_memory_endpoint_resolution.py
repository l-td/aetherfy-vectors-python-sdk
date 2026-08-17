"""
Endpoint-resolution contract for MemoryClient.

MemoryClient builds its own AetherfyVectorsClient, so it has to hand that
constructor the same inputs a direct caller would. It used to default
``endpoint`` to ``DEFAULT_ENDPOINT`` — a concrete URL, indistinguishable
downstream from a caller who asked for one. Explicit endpoints have the
highest precedence in AetherfyVectorsClient, so the default swallowed
``AETHERFY_VECTORS_URL`` entirely: a deployed Python agent that used memory
talked to the global default endpoint rather than the regional one the
control plane injects on its machine. No error, no warning.

The JS SDK never had the gap (MemoryClient spreads its config and leaves
`endpoint` undefined, so AetherfyVectorsClient's resolver runs), which is
what made this a cross-SDK behavior split rather than a shared design.

Precedence pinned here, matching AetherfyVectorsClient exactly:

    explicit endpoint=  >  AETHERFY_VECTORS_URL  >  DEFAULT_ENDPOINT

Every test deletes the env var first or sets it explicitly — an inherited
AETHERFY_VECTORS_URL in the developer's shell would otherwise make the
default-endpoint case pass or fail for the wrong reason.
"""

from unittest.mock import Mock, patch

import pytest

from aetherfy_memory import MemoryClient
from aetherfy_vectors import AetherfyVectorsClient

API_KEY = "afy_test_1234567890abcdef1234"
INJECTED_URL = "http://10.0.10.243:3000"
EXPLICIT_URL = "https://override.example.com"


def _mock_regions_response(payload, status=200):
    """Mirror of tests/test_region_param.py's helper — /api/v1/regions body."""
    import json

    resp = Mock()
    resp.status_code = status
    resp.content = json.dumps(payload).encode("utf-8")
    return resp


class TestMemoryClientEndpointResolution:
    def test_env_var_wins_when_no_endpoint_arg(self, monkeypatch):
        """The control-plane-injected URL is honored. This is the bug case."""
        monkeypatch.setenv("AETHERFY_VECTORS_URL", INJECTED_URL)

        memory = MemoryClient(api_key=API_KEY)

        assert memory.vectors.endpoint == INJECTED_URL

    def test_explicit_endpoint_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("AETHERFY_VECTORS_URL", INJECTED_URL)

        memory = MemoryClient(api_key=API_KEY, endpoint=EXPLICIT_URL)

        assert memory.vectors.endpoint == EXPLICIT_URL

    def test_default_endpoint_when_neither_is_set(self, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)

        memory = MemoryClient(api_key=API_KEY)

        assert memory.vectors.endpoint == MemoryClient.DEFAULT_ENDPOINT
        assert memory.vectors.endpoint == "https://vectors.aetherfy.com"

    def test_precedence_matches_the_vectors_client_exactly(self, monkeypatch):
        """Same three inputs through both constructors must land identically.

        Asserting parity rather than three more literals: if the resolver in
        AetherfyVectorsClient gains a step (as it did with api_region=),
        MemoryClient must inherit it rather than reimplement it.
        """
        for env_url, arg in (
            (INJECTED_URL, None),
            (INJECTED_URL, EXPLICIT_URL),
            (None, None),
            (None, EXPLICIT_URL),
        ):
            if env_url is None:
                monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
            else:
                monkeypatch.setenv("AETHERFY_VECTORS_URL", env_url)

            kwargs = {} if arg is None else {"endpoint": arg}
            memory = MemoryClient(api_key=API_KEY, **kwargs)
            direct = AetherfyVectorsClient(api_key=API_KEY, **kwargs)

            assert memory.vectors.endpoint == direct.endpoint, (
                f"divergence with env={env_url!r} arg={arg!r}: "
                f"{memory.vectors.endpoint!r} != {direct.endpoint!r}"
            )

    def test_byo_client_endpoint_is_untouched(self, monkeypatch):
        """`client=` is used as-is — the env var must not re-resolve it."""
        monkeypatch.setenv("AETHERFY_VECTORS_URL", INJECTED_URL)
        byo = AetherfyVectorsClient(api_key=API_KEY, endpoint=EXPLICIT_URL)

        memory = MemoryClient(client=byo)

        assert memory.vectors is byo
        assert memory.vectors.endpoint == EXPLICIT_URL

    @pytest.mark.parametrize("trailing", ["", "/"])
    def test_env_url_is_normalized_like_an_explicit_one(self, monkeypatch, trailing):
        monkeypatch.setenv("AETHERFY_VECTORS_URL", INJECTED_URL + trailing)

        memory = MemoryClient(api_key=API_KEY)

        assert memory.vectors.endpoint == INJECTED_URL


class TestMemoryClientApiRegionParity:
    """`api_region` reaches the resolver, and never outranks the injected URL.

    The JS SDK's MemoryClientConfig extends ClientConfig, so its MemoryClient
    has always accepted `apiRegion`. The Python MemoryClient did not take the
    kwarg at all, so the only way to select a regional endpoint through the
    memory entry point was to build an AetherfyVectorsClient by hand and pass
    it as `client=`.

    The kwarg is FORWARDED, not interpreted. That is the whole design: this
    constructor previously decided something the resolver already owned
    (defaulting `endpoint` to a concrete URL), and that decision is what
    swallowed AETHERFY_VECTORS_URL. Precedence is asserted here against
    AetherfyVectorsClient rather than re-stated, so the two entry points
    cannot drift.
    """

    def test_invalid_region_raises_at_construction(self, monkeypatch):
        """No network, no lazy failure — a typo fails where it was typed."""
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with pytest.raises(ValueError, match="api_region must be one of"):
            MemoryClient(api_key=API_KEY, api_region="us-west-2")

    def test_api_region_resolves_via_discovery(self, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            mock_get.return_value = _mock_regions_response(
                {
                    "us-east-1": "https://vectors-iad.aetherfy.run",
                    "eu-central-1": "https://vectors-fra.aetherfy.run",
                }
            )
            memory = MemoryClient(api_key=API_KEY, api_region="eu-central-1")

        assert memory.vectors.api_region == "eu-central-1"
        assert memory.vectors.endpoint == "https://vectors-fra.aetherfy.run"

    def test_injected_url_outranks_api_region(self, monkeypatch):
        """The deployed-agent case: a local-dev api_region= must not hijack it.

        Also proves no discovery call is made — a cross-ocean GET on every
        client construction would be a real cost, not just a wrong endpoint.
        """
        monkeypatch.setenv("AETHERFY_VECTORS_URL", INJECTED_URL)
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            memory = MemoryClient(api_key=API_KEY, api_region="eu-central-1")

        assert memory.vectors.endpoint == INJECTED_URL
        assert mock_get.call_count == 0

    def test_explicit_endpoint_outranks_api_region(self, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            memory = MemoryClient(
                api_key=API_KEY, endpoint=EXPLICIT_URL, api_region="eu-central-1"
            )

        assert memory.vectors.endpoint == EXPLICIT_URL
        assert mock_get.call_count == 0

    def test_env_region_is_read_when_the_kwarg_is_omitted(self, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        monkeypatch.setenv("AETHERFY_VECTORS_API_REGION", "ap-southeast-1")
        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            mock_get.return_value = _mock_regions_response(
                {"ap-southeast-1": "https://vectors-sin.aetherfy.run"}
            )
            memory = MemoryClient(api_key=API_KEY)

        assert memory.vectors.endpoint == "https://vectors-sin.aetherfy.run"

    @pytest.mark.parametrize(
        "env_url,endpoint_arg,region_arg",
        [
            (None, None, "eu-central-1"),
            (INJECTED_URL, None, "eu-central-1"),
            (None, EXPLICIT_URL, "eu-central-1"),
            (None, None, None),
        ],
    )
    def test_precedence_matches_the_vectors_client_exactly(
        self, monkeypatch, env_url, endpoint_arg, region_arg
    ):
        if env_url is None:
            monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        else:
            monkeypatch.setenv("AETHERFY_VECTORS_URL", env_url)
        monkeypatch.delenv("AETHERFY_VECTORS_API_REGION", raising=False)

        kwargs = {}
        if endpoint_arg is not None:
            kwargs["endpoint"] = endpoint_arg
        if region_arg is not None:
            kwargs["api_region"] = region_arg

        with patch("aetherfy_vectors.client.requests.get") as mock_get:
            mock_get.return_value = _mock_regions_response(
                {"eu-central-1": "https://vectors-fra.aetherfy.run"}
            )
            memory = MemoryClient(api_key=API_KEY, **kwargs)
            direct = AetherfyVectorsClient(api_key=API_KEY, **kwargs)

        assert memory.vectors.endpoint == direct.endpoint, (
            f"divergence with env={env_url!r} kwargs={kwargs!r}: "
            f"{memory.vectors.endpoint!r} != {direct.endpoint!r}"
        )
        assert memory.vectors.api_region == direct.api_region

    def test_byo_client_ignores_api_region(self, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        byo = AetherfyVectorsClient(api_key=API_KEY, endpoint=EXPLICIT_URL)

        memory = MemoryClient(client=byo, api_region="eu-central-1")

        assert memory.vectors is byo
        assert memory.vectors.endpoint == EXPLICIT_URL
