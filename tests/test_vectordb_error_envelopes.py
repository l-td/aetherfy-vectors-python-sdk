"""parse_error_response against the error bodies vectordb really authors.

The bodies in vectordb_error_envelopes.json are produced by vectordb's own code
(see ``_provenance`` in it), not typed here. vectordb puts every extra of an
error directly on the error object; this SDK keeps that whole object as
``details``, so ``field``, ``retry_after_seconds`` and the rest are
``details[...]``.
"""

import json
from pathlib import Path

from aetherfy_vectors.exceptions import (
    ConflictError,
    RateLimitExceededError,
    ValidationError,
)
from aetherfy_vectors.retry_policy import ONCE, REPEATABLE, should_retry_class
from aetherfy_vectors.utils import parse_error_response

ENVELOPES = json.loads(
    (Path(__file__).parent / "vectordb_error_envelopes.json").read_text(encoding="utf-8")
)


def test_the_fixture_carries_its_provenance_and_every_case():
    assert isinstance(ENVELOPES["_provenance"], str)
    assert sorted(ENVELOPES) == sorted(
        [
            "_provenance",
            "limit_reached_429",
            "mode_400",
            "too_soon_409",
            "unsuitable_422",
            "vectors_400",
        ]
    )


def test_400_with_field_on_the_error_object():
    error = parse_error_response(ENVELOPES["vectors_400"], 400)
    assert isinstance(error, ValidationError)
    assert error.error_code == "VALIDATION_ERROR"
    assert error.details["field"] == "vectors"


def test_400_with_field_inside_details():
    error = parse_error_response(ENVELOPES["mode_400"], 400)
    assert isinstance(error, ValidationError)
    assert error.error_code == "VALIDATION_ERROR"
    assert error.details["field"] == "storage_mode"
    assert error.details["accepted"] == ["exact", "balanced", "fast", "compact"]


def test_422_unsuitable_is_a_validation_error_with_its_code_and_extras():
    error = parse_error_response(ENVELOPES["unsuitable_422"], 422)
    assert isinstance(error, ValidationError)
    assert error.status_code == 422
    assert error.error_code == "STORAGE_MODE_UNSUITABLE"
    assert error.details["min_vector_size"] == 512
    assert error.details["suggested_mode"] == "balanced"


def test_409_too_soon_is_a_conflict_error_carrying_the_wait():
    error = parse_error_response(ENVELOPES["too_soon_409"], 409)
    assert isinstance(error, ConflictError)
    assert error.error_code == "STORAGE_MODE_CHANGE_TOO_SOON"
    assert error.details["retry_after_seconds"] == 437
    assert error.details["cooldown_seconds"] == 900


def test_429_limit_reached_carries_the_limit_and_the_reset_and_is_never_retried():
    error = parse_error_response(ENVELOPES["limit_reached_429"], 429)
    assert isinstance(error, RateLimitExceededError)
    assert error.status_code == 429
    assert error.error_code == "STORAGE_MODE_CHANGE_LIMIT_REACHED"
    assert error.details["limit"] == 3
    assert error.details["period"] == "week"
    assert error.details["resets_at"] == "2026-10-26T00:00:00Z"
    assert error.details["retry_after_seconds"] == 392000
    # The body has no `retry_after` key (and the server sends no Retry-After
    # header), so the retry policy does not treat it as a refusal worth repeating:
    # waiting inside the period cannot succeed.
    assert error.retry_after is None
    assert not should_retry_class(REPEATABLE, error)
    assert not should_retry_class(ONCE, error)
