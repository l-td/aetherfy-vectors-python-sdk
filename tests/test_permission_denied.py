"""401 and 403 are different errors.

A 401 is "the key is wrong" (AuthenticationError). A 403 is "the key is fine
and this is not allowed" (PermissionDeniedError and its subclasses). Mapped by
code first, then by status, so a 403 code the SDK does not know is still a
PermissionDeniedError, never an AuthenticationError.
"""

import pytest

import aetherfy_vectors
from aetherfy_vectors.exceptions import (
    AccessLevelInsufficientError,
    AgentKeyOutOfScopeError,
    AgentWorkspaceForbiddenError,
    AuthenticationError,
    CollectionProtectedError,
    PermissionDeniedError,
    QuotaExceededError,
)
from aetherfy_vectors.utils import parse_error_response


def body(code, **fields):
    return {"error": {"code": code, "message": f"{code} (prose)", **fields}}


CASES = [
    (
        body(
            "AGENT_ACCESS_LEVEL_INSUFFICIENT",
            workspace="alpha",
            level="read",
            required_level="read_write",
        ),
        AccessLevelInsufficientError,
        {"workspace": "alpha", "level": "read", "required_level": "read_write"},
    ),
    (
        body(
            "AGENT_ACCESS_LEVEL_INSUFFICIENT",
            workspace=None,
            level="read_write",
            required_level="full",
        ),
        AccessLevelInsufficientError,
        {"workspace": None, "level": "read_write", "required_level": "full"},
    ),
    (
        body("COLLECTION_PROTECTED", collection_name="invoices"),
        CollectionProtectedError,
        {"collection_name": "invoices"},
    ),
    (
        body("AGENT_KEY_WORKSPACE_FORBIDDEN", workspace="beta"),
        AgentWorkspaceForbiddenError,
        {"workspace": "beta"},
    ),
    (
        body("AUTH_AGENT_KEY_OUT_OF_SCOPE", agent_id="agent-1", required_scope=None),
        AgentKeyOutOfScopeError,
        {"agent_id": "agent-1", "required_scope": None},
    ),
    (body("SOME_FUTURE_REFUSAL", extra=1), PermissionDeniedError, {}),
    ({"message": "Forbidden"}, PermissionDeniedError, {}),
    (None, PermissionDeniedError, {}),
]


@pytest.mark.parametrize("payload,cls,fields", CASES)
def test_a_403_maps_by_code_then_by_status(payload, cls, fields):
    err = parse_error_response(payload, 403)
    assert type(err) is cls
    assert isinstance(err, PermissionDeniedError)
    assert not isinstance(err, AuthenticationError)
    assert err.status_code == 403
    for name, value in fields.items():
        assert getattr(err, name) == value, name


@pytest.mark.parametrize("payload,cls,fields", CASES[:7])
def test_every_403_keeps_the_servers_code_and_the_whole_body(payload, cls, fields):
    err = parse_error_response(payload, 403)
    if "error" in payload:
        assert err.error_code == payload["error"]["code"]
        for key, value in payload["error"].items():
            assert err.details[key] == value
    else:
        assert err.error_code is None
        assert err.message == "Forbidden"


def test_a_401_is_the_authentication_error_and_only_a_401():
    err = parse_error_response(body("AUTH_INVALID_KEY"), 401)
    assert type(err) is AuthenticationError
    assert not isinstance(err, PermissionDeniedError)


def test_a_plan_limit_403_is_the_quota_error_not_a_permission_error():
    err = parse_error_response(
        body("PLAN_LIMIT_EXCEEDED", limit="max_collections", allowed=3, current=3), 403
    )
    assert type(err) is QuotaExceededError
    assert not isinstance(err, PermissionDeniedError)


def test_the_classes_are_exported_from_the_package_root():
    for name in (
        "PermissionDeniedError",
        "AccessLevelInsufficientError",
        "CollectionProtectedError",
        "AgentWorkspaceForbiddenError",
        "AgentKeyOutOfScopeError",
        "QuotaExceededError",
        "InvalidFilterError",
        "InvalidRequestFieldError",
    ):
        assert getattr(aetherfy_vectors, name) is not None, name
        assert name in aetherfy_vectors.__all__, name


def test_the_specific_refusals_are_permission_errors():
    for cls in (
        AccessLevelInsufficientError,
        CollectionProtectedError,
        AgentWorkspaceForbiddenError,
        AgentKeyOutOfScopeError,
    ):
        assert issubclass(cls, PermissionDeniedError)
