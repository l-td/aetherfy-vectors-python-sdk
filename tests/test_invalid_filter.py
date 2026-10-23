"""400 INVALID_FILTER and INVALID_REQUEST_FIELD.

The API judges the keys of every filter and the members of every request
body. The answer names the key's path (``field``), the key and the nearest
valid key (``suggestion``); the SDK carries all three.
"""

from unittest.mock import Mock

import pytest

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.exceptions import (
    InvalidFilterError,
    InvalidRequestFieldError,
    ValidationError,
)
from aetherfy_vectors.utils import parse_error_response

INVALID_FILTER = {
    "error": {
        "code": "INVALID_FILTER",
        "message": "Unrecognised filter key 'mustNot' at filter.mustNot. Did you mean 'must_not'?",
        "field": "filter.mustNot",
        "key": "mustNot",
        "suggestion": "must_not",
    }
}
INVALID_REQUEST_FIELD = {
    "error": {
        "code": "INVALID_REQUEST_FIELD",
        "message": "Unrecognised request field 'fitler' at fitler. Did you mean 'filter'?",
        "field": "fitler",
        "key": "fitler",
        "suggestion": "filter",
    }
}


def test_invalid_filter_carries_path_key_and_suggestion():
    err = parse_error_response(INVALID_FILTER, 400)
    assert type(err) is InvalidFilterError
    assert isinstance(err, ValidationError)
    assert err.field == "filter.mustNot"
    assert err.key == "mustNot"
    assert err.suggestion == "must_not"
    assert err.error_code == "INVALID_FILTER"
    assert err.status_code == 400
    assert "Did you mean 'must_not'" in str(err)
    assert err.details["field"] == "filter.mustNot"


def test_a_filter_error_without_a_close_key_has_no_suggestion():
    body = {"error": dict(INVALID_FILTER["error"])}
    del body["error"]["suggestion"]
    err = parse_error_response(body, 400)
    assert type(err) is InvalidFilterError
    assert err.suggestion is None
    assert err.key == "mustNot"


def test_invalid_request_field_carries_field_and_key():
    err = parse_error_response(INVALID_REQUEST_FIELD, 400)
    assert type(err) is InvalidRequestFieldError
    assert isinstance(err, ValidationError)
    assert not isinstance(err, InvalidFilterError)
    assert err.field == "fitler"
    assert err.key == "fitler"
    assert err.suggestion == "filter"
    assert err.error_code == "INVALID_REQUEST_FIELD"


def test_a_member_the_streaming_upsert_does_not_support_carries_field_and_key():
    body = {
        "error": {
            "code": "INVALID_REQUEST_FIELD",
            "message": "Conditional upserts (update_filter, update_mode) are not supported.",
            "field": "update_filter",
            "key": "update_filter",
        }
    }
    err = parse_error_response(body, 400)
    assert type(err) is InvalidRequestFieldError
    assert (err.field, err.key, err.suggestion) == (
        "update_filter",
        "update_filter",
        None,
    )


def test_the_two_errors_are_siblings_under_validation_error():
    assert not issubclass(InvalidFilterError, InvalidRequestFieldError)
    assert not issubclass(InvalidRequestFieldError, InvalidFilterError)
    assert issubclass(InvalidFilterError, ValidationError)
    assert issubclass(InvalidRequestFieldError, ValidationError)


def test_any_other_400_is_still_a_plain_validation_error():
    err = parse_error_response(
        {"error": {"code": "VALIDATION_ERROR", "message": "x"}}, 400
    )
    assert type(err) is ValidationError


@pytest.fixture
def client(mock_requests):
    return AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="home",
    )


def _response(status, payload):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    r.content = True
    return r


def test_search_with_a_misspelt_filter_key_raises_the_typed_error(
    client, mock_requests
):
    mock_requests.request.side_effect = [_response(400, INVALID_FILTER)]
    with pytest.raises(InvalidFilterError) as exc:
        client.search("docs", [0.1, 0.2], query_filter={"mustNot": []})
    assert exc.value.suggestion == "must_not"
    assert mock_requests.request.call_count == 1  # a 400 is never retried


def test_the_request_carried_the_key_as_written(client, mock_requests):
    mock_requests.request.side_effect = [_response(400, INVALID_FILTER)]
    with pytest.raises(InvalidFilterError):
        client.search("docs", [0.1, 0.2], query_filter={"mustNot": []})
    assert mock_requests.request.call_args.kwargs["json"]["filter"] == {"mustNot": []}
