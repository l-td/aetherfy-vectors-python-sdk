"""Unit tests for create_field_index / delete_field_index.

The payload-index route has existed on the backend (and replicated) for a
while — `PUT /collections/{name}/index` and
`DELETE /collections/{name}/index/{field_name}`, both on the catch-all
allowlist and in the proxy's replicationEndpoints — but neither vectors SDK
exposed it. Filtering on an unindexed key is a SCAN, not a lookup, so a SDK
that can create a tenant key but not index it hands the customer a collection
that gets slower with every tenant.
"""

from unittest.mock import Mock, patch

import pytest
import requests

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.exceptions import (
    AetherfyVectorsException,
    RequestTimeoutError,
    ValidationError,
)

# What the server answers once the index is built (Qdrant's own shape, relayed
# by vectordb, which sends index writes with ?wait=true) ...
COMPLETED = {"result": {"operation_id": 7, "status": "completed"}, "status": "ok"}
# ... and what it answers when the build outlived its 25 s wait: HTTP 200, the
# build still running (vectordb services/proxy.js, db26396).
ACKNOWLEDGED = {
    "result": {"operation_id": None, "status": "acknowledged"},
    "status": "ok",
    "time": 25.0,
}


class TestCreateFieldIndex:
    def test_puts_field_name_and_schema_to_the_index_route(
        self, client, mock_requests, mock_successful_response
    ):
        mock_requests.request.return_value = mock_successful_response(COMPLETED)

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
        mock_requests.request.return_value = mock_successful_response(COMPLETED)
        client.create_field_index("articles", "flag", "bool")
        assert mock_requests.request.call_args.kwargs["json"]["field_schema"] == "bool"

    def test_a_parameterised_schema_object_is_forwarded_verbatim(
        self, client, mock_requests, mock_successful_response
    ):
        mock_requests.request.return_value = mock_successful_response(COMPLETED)
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
        mock_requests.request.return_value = mock_successful_response(COMPLETED)
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

    def test_a_missing_collection_is_false_not_an_exception(
        self, client, mock_requests, mock_error_response
    ):
        # The only 404 here is the collection's: a field that was never
        # indexed is answered 200 by the server, and so returns True above.
        mock_requests.request.return_value = mock_error_response(
            message="Collection articles not found",
            status_code=404,
            error_code="COLLECTION_NOT_FOUND",
        )
        assert client.delete_field_index("articles", "thread_id") is False

    def test_empty_field_name_rejected_locally(self, client, mock_requests):
        with pytest.raises(ValidationError):
            client.delete_field_index("articles", "")
        mock_requests.request.assert_not_called()


class _FakeClock:
    """Stands in for the `time` module in aetherfy_vectors.client and
    aetherfy_vectors.utils: monotonic() reads a number, sleep() adds to it.
    Nothing waits, so the assertions below are exact."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    fake = _FakeClock()
    with patch("aetherfy_vectors.client.time", fake), patch(
        "aetherfy_vectors.utils.time", fake
    ), patch("aetherfy_vectors.utils.random.random", return_value=1.0):
        yield fake


class _IndexRoute:
    """The index route on the fake clock. Each create answers the next body in
    `answers` (the last one repeats) after `cost` seconds or, when the
    attempt's timeout is shorter than that, raises requests.Timeout at the
    timeout, as the real transport would."""

    # A runaway loop ends the test instead of hanging it.
    MAX_CREATES = 1000

    def __init__(self, clock, answers, cost=25.0):
        self.clock = clock
        self.answers = list(answers)
        self.cost = cost
        self.attempts = []  # (clock time the attempt started, timeout given)

    def __call__(self, *args, **kwargs):
        given = kwargs["timeout"]
        if len(self.attempts) >= self.MAX_CREATES:
            raise AssertionError(f"more than {self.MAX_CREATES} creates")
        self.attempts.append((self.clock.now, given))
        if given < self.cost:
            self.clock.now += given
            raise requests.Timeout("fake index route: no answer yet")
        self.clock.now += self.cost
        body = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        response = Mock()
        response.status_code = 200
        response.content = True
        response.json.return_value = body
        return response


class TestCreateWaitsUntilTheIndexIsBuilt:
    """vectordb waits up to 25 s for the build, then answers "acknowledged"
    while it carries on. Returning True on that answer let an immediate
    scroll(order_by=key) fail with "No range index for order_by key"."""

    def test_acknowledged_then_completed_is_one_more_create(
        self, client, mock_requests, clock
    ):
        route = _IndexRoute(clock, [ACKNOWLEDGED, COMPLETED])
        mock_requests.request.side_effect = route

        assert client.create_field_index("articles", "ts", "integer") is True

        # The second create is the same request: re-issuing it is what waits
        # for the running build.
        assert len(route.attempts) == 2
        bodies = [c.kwargs["json"] for c in mock_requests.request.call_args_list]
        assert bodies == [{"field_name": "ts", "field_schema": "integer"}] * 2

    def test_completed_at_once_is_one_create(self, client, mock_requests, clock):
        route = _IndexRoute(clock, [COMPLETED], cost=0.01)
        mock_requests.request.side_effect = route

        assert client.create_field_index("articles", "ts", "integer") is True
        assert len(route.attempts) == 1

    def test_always_acknowledged_stops_at_the_deadline_never_true(
        self, client, mock_requests, clock
    ):
        route = _IndexRoute(clock, [ACKNOWLEDGED])
        mock_requests.request.side_effect = route

        with pytest.raises(RequestTimeoutError) as raised:
            client.create_field_index("articles", "ts", "integer", timeout=60)

        # 0 s: given 45 s (the per-create cap, under the 60 s left),
        # acknowledged at 25 s. 25 s: given the 35 s left, acknowledged at
        # 50 s. 50 s: given the last 10 s, and cut by the deadline while the
        # server is still waiting on the build. Each "acknowledged" was held
        # the server's full 25 s, so no pause was added between creates.
        assert route.attempts == [(0.0, 45.0), (25.0, 35.0), (50.0, 10.0)]
        assert clock.sleeps == []
        assert clock.now == 60.0
        message = str(raised.value)
        assert message == (
            "The payload index on 'ts' in collection 'articles' is still "
            "building after the 60 s deadline. The build carries on "
            "server-side; calling create_field_index again waits for it."
        )

    def test_a_deadline_reached_between_creates_is_still_building(
        self, client, mock_requests, clock
    ):
        # The deadline runs out as an "acknowledged" arrives, so no further
        # create is sent at all.
        route = _IndexRoute(clock, [ACKNOWLEDGED])
        mock_requests.request.side_effect = route

        with pytest.raises(RequestTimeoutError, match="still building"):
            client.create_field_index("articles", "ts", "integer", timeout=25)
        assert route.attempts == [(0.0, 25.0)]

    def test_a_deadline_that_cuts_the_first_create_does_not_claim_a_build(
        self, client, mock_requests, clock
    ):
        # No answer came back, so it is not known the create was taken.
        route = _IndexRoute(clock, [COMPLETED])
        mock_requests.request.side_effect = route

        with pytest.raises(RequestTimeoutError) as raised:
            client.create_field_index("articles", "ts", "integer", timeout=5)
        assert str(raised.value) == (
            "The payload index create on 'ts' in collection 'articles' got no "
            "answer within the 5 s deadline, so it is not known whether it was "
            "taken. Calling create_field_index again is safe."
        )
        assert route.attempts == [(0.0, 5.0)]

    @pytest.mark.parametrize(
        "body",
        [{}, {"result": True}, {"result": {"status": "clock_rejected"}}],
        ids=["no result", "bare true", "another status"],
    )
    def test_any_other_answer_raises_rather_than_returning_true(
        self, client, mock_requests, clock, body
    ):
        route = _IndexRoute(clock, [body], cost=0.01)
        mock_requests.request.side_effect = route

        with pytest.raises(AetherfyVectorsException, match="not confirmed built"):
            client.create_field_index("articles", "ts", "integer")
        assert len(route.attempts) == 1


class TestCreateIsNeverUnbounded:
    """A server that answers "acknowledged" at once did not hold the create
    (a vectordb from before db26396 answers every create that way). Re-sending
    at once would hammer it, forever when no timeout was passed."""

    def test_an_instant_acknowledged_is_paced_and_ends_at_the_default_deadline(
        self, client, mock_requests, clock
    ):
        route = _IndexRoute(clock, [ACKNOWLEDGED], cost=0.05)
        mock_requests.request.side_effect = route

        with pytest.raises(RequestTimeoutError) as raised:
            client.create_field_index("articles", "ts", "integer")

        # Pauses of 1, 2, 4, 8, then 10 s each: 9 creates start in the first
        # minute. With no floor it would be 1200 (one per 0.05 s).
        first_minute = [at for at, _ in route.attempts if at < 60.0]
        assert len(first_minute) == 9, route.attempts[:12]
        assert clock.sleeps[:6] == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]
        assert max(clock.sleeps) == 10.0
        # And the call ends at the default deadline, not never.
        assert clock.now == AetherfyVectorsClient.INDEX_DEFAULT_DEADLINE_S
        assert len(route.attempts) < 70
        assert str(raised.value) == (
            "The payload index on 'ts' in collection 'articles' is still "
            "building after the 600 s deadline. The build carries on "
            "server-side; calling create_field_index again waits for it."
        )

    def test_a_server_that_holds_each_create_gets_no_added_pause(
        self, client, mock_requests, clock
    ):
        route = _IndexRoute(
            clock, [ACKNOWLEDGED, ACKNOWLEDGED, ACKNOWLEDGED, COMPLETED]
        )
        mock_requests.request.side_effect = route

        assert client.create_field_index("articles", "ts", "integer") is True
        assert clock.sleeps == []
        assert [at for at, _ in route.attempts] == [0.0, 25.0, 50.0, 75.0]

    def test_the_default_deadline_is_ten_minutes(self):
        assert AetherfyVectorsClient.INDEX_DEFAULT_DEADLINE_S == 600.0


class TestCreateAttemptTimeout:
    """One create may be held for the server's whole wait budget, plus a
    forward to a hosting region. An HTTP timeout shorter than that gives up
    before the server's own answer arrives."""

    def test_the_attempt_timeout_outlasts_the_server_wait_and_a_forward(self):
        # INDEX_WAIT_BUDGET_S mirrors vectordb backend/config/timeouts.js
        # INDEX_WAIT_BUDGET_MS, and INDEX_FORWARD_MARGIN_S its
        # FORWARD_MARGIN_MS. No cross-repo gate reads those; the comment on
        # the constants says to copy a change.
        C = AetherfyVectorsClient
        assert C.INDEX_WAIT_BUDGET_S == 25.0
        assert C.INDEX_FORWARD_MARGIN_S == 5.0
        # With room for this client's own hop on top.
        held = C.INDEX_WAIT_BUDGET_S + C.INDEX_FORWARD_MARGIN_S
        assert C.INDEX_CREATE_ATTEMPT_TIMEOUT_S >= held + 10.0
        # The client-wide default alone does not leave that room, which is
        # why the create does not use it.
        assert C.DEFAULT_TIMEOUT < held + 10.0

    def test_each_create_without_a_deadline_gets_the_index_attempt_timeout(
        self, client, mock_requests, clock
    ):
        # The fixture client's own timeout is 10 s.
        route = _IndexRoute(clock, [ACKNOWLEDGED, COMPLETED])
        mock_requests.request.side_effect = route

        client.create_field_index("articles", "ts", "integer")
        assert [given for _, given in route.attempts] == [
            AetherfyVectorsClient.INDEX_CREATE_ATTEMPT_TIMEOUT_S
        ] * 2

    def test_a_longer_client_timeout_is_kept(
        self, api_key, test_endpoint, mock_requests, clock
    ):
        patient = AetherfyVectorsClient(
            api_key=api_key, endpoint=test_endpoint, timeout=90.0
        )
        route = _IndexRoute(clock, [COMPLETED])
        mock_requests.request.side_effect = route

        patient.create_field_index("articles", "ts", "integer")
        assert [given for _, given in route.attempts] == [90.0]
