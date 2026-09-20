"""Unit tests for create_field_index / delete_field_index.

The payload-index route has existed on the backend (and replicated) for a
while — `PUT /collections/{name}/index` and
`DELETE /collections/{name}/index/{field_name}`, both on the catch-all
allowlist and in the proxy's replicationEndpoints — but neither vectors SDK
exposed it. Filtering on an unindexed key is a SCAN, not a lookup, so a SDK
that can create a tenant key but not index it hands the customer a collection
that gets slower with every tenant.
"""

import pytest

from aetherfy_vectors.exceptions import ValidationError


class TestCreateFieldIndex:
    def test_puts_field_name_and_schema_to_the_index_route(
        self, client, mock_requests, mock_successful_response
    ):
        mock_requests.request.return_value = mock_successful_response({})

        assert client.create_field_index("articles", "thread_id") is True

        kwargs = mock_requests.request.call_args.kwargs
        assert kwargs["method"] == "PUT"
        assert kwargs["url"].endswith("/collections/articles/index")
        assert kwargs["json"] == {
            "field_name": "thread_id",
            "field_schema": "keyword",
        }

    def test_schema_is_forwarded_verbatim(
        self, client, mock_requests, mock_successful_response
    ):
        mock_requests.request.return_value = mock_successful_response({})
        client.create_field_index("articles", "flag", "bool")
        assert mock_requests.request.call_args.kwargs["json"]["field_schema"] == "bool"

    def test_a_parameterised_schema_object_is_forwarded_verbatim(
        self, client, mock_requests, mock_successful_response
    ):
        mock_requests.request.return_value = mock_successful_response({})
        schema = {"type": "text", "tokenizer": "word", "lowercase": True}
        client.create_field_index("articles", "body", schema)
        assert mock_requests.request.call_args.kwargs["json"]["field_schema"] == schema

    def test_workspace_routing_uses_the_nested_url(
        self, api_key, test_endpoint, mock_requests, mock_successful_response
    ):
        from aetherfy_vectors import AetherfyVectorsClient

        ws_client = AetherfyVectorsClient(
            api_key=api_key, endpoint=test_endpoint, workspace="team-alpha"
        )
        mock_requests.request.return_value = mock_successful_response({})
        ws_client.create_field_index("articles", "thread_id")
        url = mock_requests.request.call_args.kwargs["url"]
        assert url.endswith("/workspaces/team-alpha/collections/articles/index")

    def test_empty_field_name_rejected_locally(self, client, mock_requests):
        with pytest.raises(ValidationError):
            client.create_field_index("articles", "")
        mock_requests.request.assert_not_called()

    def test_invalid_collection_name_rejected_locally(self, client, mock_requests):
        with pytest.raises(ValidationError):
            client.create_field_index("bad/name", "thread_id")
        mock_requests.request.assert_not_called()


class TestDeleteFieldIndex:
    def test_deletes_the_field_scoped_route(
        self, client, mock_requests, mock_successful_response
    ):
        mock_requests.request.return_value = mock_successful_response({})

        assert client.delete_field_index("articles", "thread_id") is True

        kwargs = mock_requests.request.call_args.kwargs
        assert kwargs["method"] == "DELETE"
        assert kwargs["url"].endswith("/collections/articles/index/thread_id")

    def test_dotted_field_name_is_url_encoded(
        self, client, mock_requests, mock_successful_response
    ):
        mock_requests.request.return_value = mock_successful_response({})
        client.delete_field_index("articles", "metadata/tag")
        url = mock_requests.request.call_args.kwargs["url"]
        # The field name is one path segment; a '/' inside it must not become
        # a separator, or the request lands on a path the allowlist rejects.
        assert url.endswith("/collections/articles/index/metadata%2Ftag")

    def test_missing_index_is_false_not_an_exception(
        self, client, mock_requests, mock_error_response
    ):
        mock_requests.request.return_value = mock_error_response(
            message="Index not found", status_code=404, error_code="NOT_FOUND"
        )
        assert client.delete_field_index("articles", "thread_id") is False

    def test_empty_field_name_rejected_locally(self, client, mock_requests):
        with pytest.raises(ValidationError):
            client.delete_field_index("articles", "")
        mock_requests.request.assert_not_called()
