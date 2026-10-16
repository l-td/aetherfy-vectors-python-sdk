"""An exception built from a server's answer carries the server's code.

Found by the live e2e: delete_schema caught the server's 404 and raised a fresh
SchemaNotFoundError whose error_code was None (the server said
SCHEMA_NOT_DEFINED). Code that branches on ``e.error_code`` never matched.

Two kinds of test: the sites that rebuild an exception are driven with the
contract sheet's bodies, and a rule reads the package and fails when an
``except`` block raises an SDK exception that does not pass a code, unless the
case is listed with its reason.
"""

import ast
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.exceptions import (
    AetherfyVectorsException,
    SchemaNotFoundError,
    ValidationError,
)
from aetherfy_vectors.models import Point

PACKAGE = Path(__file__).resolve().parent.parent / "aetherfy_vectors"


def _response(status, code, **fields):
    r = Mock()
    r.status_code = status
    r.json.return_value = {"error": {"code": code, "message": f"{code} (prose)", **fields}}
    r.content = True
    return r


@pytest.fixture
def client():
    c = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="w",
    )
    c.session.request = Mock()
    return c


# ---------------------------------------------------------------------------
# The reported bug, and its sibling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["SCHEMA_NOT_DEFINED", "COLLECTION_NOT_FOUND"])
def test_delete_schema_not_found_carries_the_servers_code(client, code):
    client.session.request.side_effect = [
        _response(404, code, collection_name="docs")
    ]

    with pytest.raises(SchemaNotFoundError) as exc:
        client.delete_schema("docs")

    assert exc.value.error_code == code
    assert exc.value.status_code == 404
    assert isinstance(exc.value.__cause__, AetherfyVectorsException)


def _chunk_client(client, first_error, then=None):
    """A client whose PUT raises `first_error`, then `then` (default: the same
    server error again, so the 412 retry fails)."""
    calls = []

    def make_request(method, endpoint, *args, **kwargs):
        calls.append(method)
        if len(calls) == 1:
            raise first_error
        raise (then or first_error)

    client._make_request = make_request
    client.get_schema = Mock(return_value=None)
    client._fetch_and_cache_schema = Mock(return_value={"etag": "v2"})
    return client


def _server_error(cls, status, code):
    return cls("x", status_code=status, error_code=code, request_id="req-1")


def test_a_failed_412_retry_carries_the_original_412_code(client):
    """Upsert, 412 (the schema changed), the retry fails too: the error that
    is raised says so and carries the 412's SCHEMA_VERSION_MISMATCH."""
    mismatch = _server_error(ValidationError, 412, "SCHEMA_VERSION_MISMATCH")
    _chunk_client(client, mismatch, then=RuntimeError("retry failed"))

    with pytest.raises(ValidationError) as exc:
        client._upload_points_chunk("w/docs", "docs", [{"id": 1, "vector": [0.1]}], {}, None)

    assert exc.value.status_code == 412
    assert exc.value.error_code == "SCHEMA_VERSION_MISMATCH"
    assert exc.value.request_id == "req-1"
    assert exc.value.__cause__ is mismatch


def test_a_single_chunk_400_is_still_a_value_error_with_the_server_error_as_cause(client):
    rejected = _server_error(ValidationError, 400, "VALIDATION_ERROR")
    _chunk_client(client, rejected)

    with pytest.raises(ValueError) as exc:
        client._upload_points_chunk("w/docs", "docs", [{"id": 1, "vector": [0.1]}], {}, None)

    assert not isinstance(exc.value, AetherfyVectorsException)  # its documented type
    assert exc.value.__cause__.error_code == "VALIDATION_ERROR"


def test_a_rejected_chunk_in_a_partial_upsert_keeps_the_servers_code(client):
    """In a PartialUpsertError's failed list the ValidationError for a chunk
    the server rejected (400) carries the server's code."""
    from aetherfy_vectors.exceptions import PartialUpsertError

    rejected = _server_error(ValidationError, 400, "VALIDATION_ERROR")
    calls = []

    def upload(scoped, name, chunk, schema, payload_schema):
        calls.append(chunk)
        if len(calls) == 1:
            try:
                raise rejected
            except ValidationError as e:
                raise ValueError(str(e)) from e
        return True

    client._upload_points_chunk = upload
    client._fetch_and_cache_schema = Mock(return_value={"etag": "v1"})
    client._get_cached_schema = Mock(return_value={"size": 2, "distance": "Cosine", "etag": "v1"})
    client._payload_schema_cache[client._scope_collection("docs")] = {"schema": None, "etag": None}
    # Force two chunks.
    with patch("aetherfy_vectors.client.chunk_points_by_bytes", lambda pts, *a, **k: [pts[:1], pts[1:]]):
        with pytest.raises(PartialUpsertError) as exc:
            client.upsert("docs", [Point(id=1, vector=[0.1, 0.2]), Point(id=2, vector=[0.1, 0.2])])

    (failed,) = exc.value.failed
    assert failed["error"].error_code == "VALIDATION_ERROR"
    assert failed["error"].status_code == 400


def test_region_discovery_non_200_carries_status_and_code():
    answer = Mock(status_code=503, content=True)
    answer.json.return_value = {
        "error": {"code": "WORKSPACE_LOOKUP_UNAVAILABLE", "message": "x"}
    }
    with patch("aetherfy_vectors.client.requests.get", return_value=answer):
        with pytest.raises(AetherfyVectorsException) as exc:
            AetherfyVectorsClient(
                api_key="afy_test_1234567890abcdef1234",
                workspace="w",
                api_region="eu-central-1",
            )
    assert exc.value.status_code == 503
    assert exc.value.error_code == "WORKSPACE_LOOKUP_UNAVAILABLE"


# ---------------------------------------------------------------------------
# The rule: no except block replaces a server error with a code-less one
# ---------------------------------------------------------------------------

# (function, exception) raised inside an `except` WITHOUT a server code, and
# why that is right: the exception is not built from a server answer.
NO_SERVER_ANSWER = {
    ("make_single_request", "RequestTimeoutError"): "a timeout: no response",
    ("make_single_request", "NetworkError"): "a connection failure: no response",
    ("make_single_request", "AetherfyVectorsException"): "a requests error: no response",
    ("_resolve_region_endpoint", "AetherfyVectorsException"): (
        "discovery's transport and non-JSON failures; a non-200 answer is "
        "raised outside an except and carries its code (tested above)"
    ),
    ("_upload_points_chunk", "SchemaValidationError"): (
        "the client's own validation of the refreshed schema, not a server answer"
    ),
}


def _unguarded_raises():
    """(function, exception, line) for each `raise SDKError(...)` inside an
    except block of client.py that passes no error_code and is not listed."""
    tree = ast.parse((PACKAGE / "client.py").read_text(encoding="utf-8"))
    found = []

    def passes_code(call):
        return any(
            kw.arg == "error_code" or kw.arg is None  # **server_fields(...)
            for kw in call.keywords
        )

    class Walker(ast.NodeVisitor):
        func = None
        in_except = 0

        def visit_FunctionDef(self, node):
            old, self.func = self.func, node.name
            self.generic_visit(node)
            self.func = old

        def visit_ExceptHandler(self, node):
            self.in_except += 1
            self.generic_visit(node)
            self.in_except -= 1

        def visit_Raise(self, node):
            exc = node.exc
            if self.in_except and isinstance(exc, ast.Call):
                name = ast.unparse(exc.func)
                is_sdk = name.endswith(("Error", "Exception")) and name not in (
                    "TypeError",
                    "ValueError",
                    "RuntimeError",
                    "KeyError",
                )
                if (
                    is_sdk
                    and not passes_code(exc)
                    and (self.func, name) not in NO_SERVER_ANSWER
                ):
                    found.append((self.func, name, node.lineno))
            self.generic_visit(node)

    Walker().visit(tree)
    return found


def test_no_except_block_raises_an_sdk_error_without_the_servers_code():
    assert _unguarded_raises() == []


def test_the_exception_list_has_no_stale_entries():
    """Every listed (function, exception) still exists in an except block."""
    tree = ast.parse((PACKAGE / "client.py").read_text(encoding="utf-8"))
    raised = set()

    class Walker(ast.NodeVisitor):
        func = None
        in_except = 0

        def visit_FunctionDef(self, node):
            old, self.func = self.func, node.name
            self.generic_visit(node)
            self.func = old

        def visit_ExceptHandler(self, node):
            self.in_except += 1
            self.generic_visit(node)
            self.in_except -= 1

        def visit_Raise(self, node):
            if self.in_except and isinstance(node.exc, ast.Call):
                raised.add((self.func, ast.unparse(node.exc.func)))
            self.generic_visit(node)

    Walker().visit(tree)
    assert set(NO_SERVER_ANSWER) <= raised, set(NO_SERVER_ANSWER) - raised
