"""Unit tests for client.set_payload / overwrite_payload / delete_payload.

Pins:
  - HTTP method + path + body shape match the WS3 dashboard proxies.
  - Validators (collection name, point IDs) fire before the request.
"""

from unittest.mock import MagicMock, patch

import pytest

from aetherfy_vectors.client import AetherfyVectorsClient
from aetherfy_vectors.exceptions import ValidationError


def _make_client():
    client = AetherfyVectorsClient.__new__(AetherfyVectorsClient)
    client.workspace = None
    client._make_request = MagicMock(return_value={"result": {"status": "ok"}})
    return client


# ---------- set_payload -----------------------------------------------------


def test_set_payload_posts_to_payload_endpoint_with_correct_body():
    client = _make_client()
    out = client.set_payload("col", {"tag": "new"}, ["p1", "p2"])

    method, path = client._make_request.call_args.args[:2]
    body = client._make_request.call_args.args[2]
    assert method == "POST"
    assert path == "collections/col/points/payload"
    assert body == {"payload": {"tag": "new"}, "points": ["p1", "p2"]}
    assert out == {"status": "ok"}


# ---------- overwrite_payload ----------------------------------------------


def test_overwrite_payload_uses_put():
    client = _make_client()
    client.overwrite_payload("col", {"only": "this"}, ["p1"])

    method, path = client._make_request.call_args.args[:2]
    body = client._make_request.call_args.args[2]
    assert method == "PUT"
    assert path == "collections/col/points/payload"
    assert body == {"payload": {"only": "this"}, "points": ["p1"]}


# ---------- delete_payload --------------------------------------------------


def test_delete_payload_uses_delete_with_keys_and_points_body():
    client = _make_client()
    client.delete_payload("col", ["old_tag", "old_meta"], ["p1", "p2"])

    method, path = client._make_request.call_args.args[:2]
    body = client._make_request.call_args.args[2]
    assert method == "DELETE"
    assert path == "collections/col/points/payload"
    assert body == {"keys": ["old_tag", "old_meta"], "points": ["p1", "p2"]}


# ---------- validators ------------------------------------------------------


def test_set_payload_validates_collection_name_first():
    client = _make_client()
    with patch("aetherfy_vectors.client.validate_collection_name") as mock_v:
        mock_v.side_effect = ValidationError("bad name")
        with pytest.raises(ValidationError):
            client.set_payload("col", {"a": 1}, ["p1"])
        mock_v.assert_called_once_with("col")
    # Request never fires when validation throws.
    assert client._make_request.call_count == 0


def test_overwrite_payload_validates_each_point_id():
    client = _make_client()
    with patch("aetherfy_vectors.client.validate_point_id") as mock_v:
        client.overwrite_payload("col", {"x": 1}, ["p1", 42])
        # validate_point_id is called once per id.
        assert mock_v.call_args_list[0].args[0] == "p1"
        assert mock_v.call_args_list[1].args[0] == 42


def test_delete_payload_validates_each_point_id():
    client = _make_client()
    with patch("aetherfy_vectors.client.validate_point_id") as mock_v:
        client.delete_payload("col", ["k"], ["p1", "p2", "p3"])
        assert mock_v.call_count == 3
