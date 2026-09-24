"""
A 409 with no more specific class raises ConflictError.

Parity with the JavaScript SDK, which has always mapped the generic 409 to
`ConflictError`; Python used to hand back the bare `AetherfyVectorsException`.
The two 409s that carry typed fields keep their own classes, and a handler
written against the base must see no difference.
"""

from unittest.mock import Mock

import pytest

from aetherfy_vectors import (
    AetherfyVectorsException,
    CollectionInOtherRegionError,
    CollectionInUseError,
    ConflictError,
)
from aetherfy_vectors.utils import parse_error_response


def _respond_409(mock_requests, body):
    response = Mock()
    response.status_code = 409
    response.json.return_value = body
    response.content = True
    mock_requests.request.return_value = response


class TestGeneric409:
    def test_unknown_code_is_conflict_error(self):
        body = {"error": {"code": "SOMETHING_ELSE", "message": "nope"}}
        err = parse_error_response(body, 409)
        assert type(err) is ConflictError
        assert err.status_code == 409
        assert err.error_code == "SOMETHING_ELSE"
        assert err.message == "nope"

    def test_flat_and_unstructured_bodies_are_conflict_error(self):
        # Every body shape parse_error_response accepts reaches the same
        # class: the mapping is on the status, the code only refines it.
        for body in ({"message": "flat"}, "a bare string", None):
            err = parse_error_response(body, 409)
            assert type(err) is ConflictError, body
            assert err.status_code == 409

    def test_client_raises_conflict_error(self, client, mock_requests):
        _respond_409(
            mock_requests,
            {"error": {"code": "SOMETHING_ELSE", "message": "Resource conflict"}},
        )
        with pytest.raises(ConflictError) as exc_info:
            client.delete_collection("test-collection")
        assert exc_info.value.error_code == "SOMETHING_ELSE"


class TestSpecific409sKeepTheirClasses:
    def test_collection_in_use(self):
        body = {
            "error": {
                "code": "COLLECTION_IN_USE",
                "message": "in use",
                "collection_name": "docs",
                "agents": ["bot"],
            }
        }
        err = parse_error_response(body, 409)
        assert type(err) is CollectionInUseError
        assert not isinstance(err, ConflictError)
        assert err.agents == ["bot"]

    def test_collection_exists_in_other_region(self):
        body = {
            "error": {
                "code": "COLLECTION_EXISTS_IN_OTHER_REGION",
                "message": "elsewhere",
                "collection_name": "docs",
                "existing_regions": ["eu-central-1"],
                "requesting_region": "us-east-1",
            }
        }
        err = parse_error_response(body, 409)
        assert type(err) is CollectionInOtherRegionError
        assert not isinstance(err, ConflictError)
        assert err.existing_regions == ["eu-central-1"]


class TestBaseHandlersAreUnchanged:
    def test_conflict_error_is_caught_as_the_base(self, client, mock_requests):
        _respond_409(mock_requests, {"error": {"code": "X", "message": "m"}})
        caught = None
        try:
            client.delete_collection("test-collection")
        except AetherfyVectorsException as e:
            caught = e
        assert isinstance(caught, ConflictError)
        assert caught.status_code == 409
