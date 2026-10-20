"""An agent's key refused a workspace it was not granted arrives as a typed
error that names the workspace and how to grant access -- through the one
status/code mapping every error goes through (utils.parse_error_response).

The body is the vector API's (vectordb services/agentAccess.js decideScope).
"""
import pytest

from aetherfy_vectors import AetherfyVectorsException, AgentWorkspaceForbiddenError
from aetherfy_vectors.exceptions import ServiceUnavailableError
from aetherfy_vectors.utils import parse_error_response


def _forbidden(workspace):
    return {
        "error": {
            "code": "AGENT_KEY_WORKSPACE_FORBIDDEN",
            "message": "This agent's key cannot use ...",
            "workspace": workspace,
            "documentation_url": "https://docs.aetherfy.com/platform/api-keys",
        }
    }


def test_a_forbidden_workspace_is_typed_and_names_it():
    err = parse_error_response(_forbidden("beta"), 403)
    assert isinstance(err, AgentWorkspaceForbiddenError)
    assert isinstance(err, AetherfyVectorsException)
    assert err.workspace == "beta"
    assert err.status_code == 403
    assert err.error_code == "AGENT_KEY_WORKSPACE_FORBIDDEN"
    assert "workspace 'beta'" in str(err)
    assert "afy access <agent> --add beta" in str(err)


def test_a_workspace_addressed_by_id_is_this_workspace():
    """Addressed by id (workspace="auto"), the body's `workspace` is null."""
    err = parse_error_response(_forbidden(None), 403)
    assert isinstance(err, AgentWorkspaceForbiddenError)
    assert err.workspace is None
    assert "this workspace" in str(err)
    assert "no workspace" not in str(err)
    assert "afy access <agent> --add <workspace>" in str(err)


def test_control_another_403_is_not_this_error():
    err = parse_error_response(
        {"error": {"code": "AUTH_AGENT_KEY_OUT_OF_SCOPE", "message": "no", "agent_id": "a"}}, 403
    )
    assert not isinstance(err, AgentWorkspaceForbiddenError)
    assert err.error_code == "AUTH_AGENT_KEY_OUT_OF_SCOPE"


def test_an_unreadable_allowed_set_is_the_retryable_503():
    err = parse_error_response(
        {"error": {"code": "AGENT_ACCESS_UNAVAILABLE", "message": "Retry."}}, 503
    )
    assert isinstance(err, ServiceUnavailableError)
    assert err.error_code == "AGENT_ACCESS_UNAVAILABLE"


@pytest.mark.parametrize("status", [400, 404, 409])
def test_control_the_code_under_another_status_is_not_mapped(status):
    assert not isinstance(parse_error_response(_forbidden("beta"), status), AgentWorkspaceForbiddenError)
