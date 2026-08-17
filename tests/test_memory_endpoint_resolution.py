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

import pytest

from aetherfy_memory import MemoryClient
from aetherfy_vectors import AetherfyVectorsClient

API_KEY = "afy_test_1234567890abcdef1234"
INJECTED_URL = "http://10.0.10.243:3000"
EXPLICIT_URL = "https://override.example.com"


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
