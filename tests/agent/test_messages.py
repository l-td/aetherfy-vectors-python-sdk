"""
Messages between running agents: send, receive, the history, restarted() and parent().

What each call SENDS is read from the Transport stub (tests/agent/conftest.py), and what the control plane answers is
queued on it, in the shapes its routes answer with (POST /deployments/{id}/messages, POST .../messages/receive,
GET .../messages). The local wait on the supervisor goes through the same transport, at AETHERFY_SUPERVISOR_WAIT_URL.
"""

import uuid

import pytest

import aetherfy_agent
from aetherfy_agent import (
    AgentTransportError,
    Ended,
    MailboxFull,
    Message,
    MessageError,
    MessageRateLimited,
    MessageStorageFull,
    MessageTooLarge,
    NotRunningOnAgent,
    RecipientEnded,
    RunAccessDenied,
    RunNotFound,
    SenderEnded,
    Sent,
    messages,
    parent,
    receive,
    restarted,
    send,
    service_run,
)

API = "https://agents.aetherfy.com/api/v1"
OWN = "11111111-1111-4111-8111-111111111111"
PARENT = "22222222-2222-4222-8222-222222222222"
CHILD = "33333333-3333-4333-8333-333333333333"
LOCAL = "http://127.0.0.1:9400"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", API)
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_run_key")
    # The mailbox is found through AETHERFY_DEPLOYMENT_ID alone: the run on a task machine, the release on a
    # service's. AETHERFY_SPAWN_ID names a run only, so it is absent here, as on a service's machine.
    monkeypatch.setenv("AETHERFY_DEPLOYMENT_ID", OWN)
    monkeypatch.delenv("AETHERFY_SPAWN_ID", raising=False)
    monkeypatch.setenv("AETHERFY_SUPERVISOR_WAIT_URL", LOCAL)
    aetherfy_agent._pending_acks.clear()
    aetherfy_agent._pending_waits.clear()
    yield
    aetherfy_agent._pending_acks.clear()
    aetherfy_agent._pending_waits.clear()


def queued(transport, *replies):
    """Answer each request with the next reply, in order; record nothing else."""
    pending = list(replies)

    def route(method, url):
        assert pending, "a request was made that the test did not expect: {0} {1}".format(method, url)
        reply = pending.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    transport.route = route
    return pending


def stored(seq, *, sender_run_id=PARENT, body=None):
    return {
        "id": "m-{0}".format(seq),
        "seq": seq,
        "sender": "run",
        "sender_run_id": sender_run_id,
        "body": body if body is not None else {"n": seq},
        "created_at": "2026-11-18T12:00:00Z",
    }


# --- SEND -----------------------------------------------------------------------


def test_send_posts_the_message_with_an_id_of_its_own(transport):
    queued(transport, (201, {"message": {"id": "x", "seq": 3}, "recipient": {"id": PARENT, "agent": "boss", "state": "awaiting_message"}}))

    sent = send(PARENT, {"progress": 0.5})

    call = transport.calls[0]
    assert (call["method"], call["url"]) == ("POST", "{0}/deployments/{1}/messages".format(API, PARENT))
    assert call["body"]["body"] == {"progress": 0.5}
    assert str(uuid.UUID(call["body"]["id"])) == call["body"]["id"]
    # REPEATABLE: the id makes a resend the same message, stored once.
    assert call["kwargs"]["repeatable"] is True
    assert sent == Sent(message_id="x", seq=3, recipient_id=PARENT, recipient_agent="boss", recipient_state="awaiting_message")


def test_send_keeps_an_id_it_is_given(transport):
    queued(transport, (200, {"message": {"id": "mine", "seq": 1}, "recipient": {"id": PARENT}}))
    send(PARENT, "hello", id="mine")
    assert transport.calls[0]["body"]["id"] == "mine"


@pytest.mark.parametrize(
    "status, code, error",
    [
        (413, "RUN_MESSAGE_TOO_LARGE", MessageTooLarge),
        (429, "RUN_MESSAGE_RATE_LIMITED", MessageRateLimited),
        (429, "RUN_MAILBOX_FULL", MailboxFull),
        (429, "RUN_MESSAGE_STORAGE_FULL", MessageStorageFull),
        (409, "RUN_MESSAGE_RECIPIENT_ENDED", RecipientEnded),
        (404, "DEPLOYMENT_NOT_FOUND", RunNotFound),
        (403, "AUTH_AGENT_KEY_OUT_OF_SCOPE", RunAccessDenied),
        (409, "SOMETHING_NEW", MessageError),
    ],
)
def test_a_refused_send_raises_by_its_code(transport, status, code, error):
    queued(transport, (status, {"detail": {"code": code, "message": "no"}}))
    with pytest.raises(error) as raised:
        send(PARENT, "hello")
    assert raised.value.error_code == code


def test_a_rate_limit_says_when_to_send_again(transport):
    transport.retry_after = 12.0
    queued(transport, (429, {"detail": {"code": "RUN_MESSAGE_RATE_LIMITED", "message": "slow down"}}))
    with pytest.raises(MessageRateLimited) as raised:
        send(PARENT, "hello")
    assert raised.value.retry_after == 12.0


# --- RECEIVE --------------------------------------------------------------------


def test_receive_hands_over_the_oldest_unread_and_acknowledges_it_on_the_next_call(transport):
    queued(transport, (200, {"message": stored(1)}), (200, {"message": stored(2)}))

    first = receive()
    second = receive()

    assert isinstance(first, Message) and (first.seq, first.body, first.sender_run_id) == (1, {"n": 1}, PARENT)
    assert second.seq == 2
    receives = [c for c in transport.calls if c["url"].endswith("/messages/receive")]
    assert receives[0]["url"] == "{0}/deployments/{1}/messages/receive".format(API, OWN)
    assert receives[0]["body"]["ack"] == []
    # AT LEAST ONCE: a message is acknowledged on the NEXT call, never on the answer that handed it over.
    assert receives[1]["body"]["ack"] == ["m-1"]


def test_an_unacknowledged_message_stays_pending_when_the_next_call_fails(transport):
    queued(transport, (200, {"message": stored(1)}), (503, {"detail": {"code": "SERVICE_UNAVAILABLE", "message": "x"}}),
           (200, {"waiting": False}))
    receive()
    with pytest.raises(MessageError):
        receive(timeout_seconds=0)
    receive(timeout_seconds=0)
    receives = [c for c in transport.calls if c["url"].endswith("/messages/receive")]
    assert receives[2]["body"]["ack"] == ["m-1"]


def test_a_check_with_nothing_to_read_returns_none_and_waits_for_nothing(transport):
    queued(transport, (200, {"waiting": False}))
    assert receive(timeout_seconds=0) is None
    assert transport.calls[0]["body"]["timeout_seconds"] == 0
    assert len(transport.calls) == 1


def test_a_wait_blocks_on_the_supervisors_local_call_then_asks_once_more(transport):
    queued(
        transport,
        (200, {"waiting": True, "wait_id": "w-1"}),
        (200, {"released": "message"}),
        (200, {"message": stored(4)}),
    )

    got = receive()

    assert got.seq == 4
    local = transport.calls[1]
    # LOCAL: the wait is held by this machine's supervisor, never by a request to the control plane.
    assert (local["method"], local["url"]) == ("POST", LOCAL + "/wait")
    assert local["body"] == {"wait_id": "w-1"}
    assert local["kwargs"]["timeout"] is None
    assert local["api_key"] == "afy_test_run_key"


def test_a_wait_released_by_its_deadline_returns_none(transport):
    queued(transport, (200, {"waiting": True, "wait_id": "w-2"}), (200, {"released": "timeout"}))
    assert receive(timeout_seconds=30) is None
    assert transport.calls[0]["body"]["timeout_seconds"] == 30


def test_a_childs_end_is_handed_over_and_acknowledged_by_its_run_id(transport):
    queued(transport, (200, {"ended": {"run_id": CHILD, "state": "completed"}}), (200, {"waiting": False}))
    got = receive()
    assert isinstance(got, Ended) and (got.run_id, got.state) == (CHILD, "completed")
    receive(timeout_seconds=0)
    assert transport.calls[1]["body"]["ack"] == [CHILD]


def test_from_run_is_sent_and_an_ended_sender_raises(transport):
    queued(transport, (409, {"detail": {"code": "RUN_MESSAGE_SENDER_ENDED", "message": "ended", "sender_state": "failed"}}))
    with pytest.raises(SenderEnded) as raised:
        receive(from_run=CHILD)
    assert transport.calls[0]["body"]["from_run"] == CHILD
    assert raised.value.sender_state == "failed"


@pytest.mark.parametrize("timeout", [None, 30])
def test_without_the_supervisors_wait_a_wait_is_refused_before_any_request(transport, monkeypatch, timeout):
    # The request would record the wait and put the run to sleep after the grace, under a process not waiting.
    monkeypatch.delenv("AETHERFY_SUPERVISOR_WAIT_URL")
    queued(transport)
    with pytest.raises(NotRunningOnAgent):
        receive(timeout_seconds=timeout)
    assert transport.calls == []


RELEASE = "44444444-4444-4444-8444-444444444444"


def test_a_service_reads_its_releases_mailbox_with_no_spawn_id(transport, monkeypatch):
    # A service's machine: AETHERFY_DEPLOYMENT_ID is the release it serves, and there is no AETHERFY_SPAWN_ID.
    monkeypatch.setenv("AETHERFY_DEPLOYMENT_ID", RELEASE)
    queued(transport, (200, {"message": stored(1, sender_run_id=CHILD, body={"question": "?"})}))
    got = receive(timeout_seconds=0, from_run=CHILD)
    assert (got.body, got.sender_run_id) == ({"question": "?"}, CHILD)
    assert transport.calls[0]["url"] == "{0}/deployments/{1}/messages/receive".format(API, RELEASE)


@pytest.mark.parametrize("call", [lambda: receive(timeout_seconds=0), lambda: list(messages())])
def test_the_mailbox_is_never_read_from_aetherfy_spawn_id(transport, monkeypatch, call):
    # One source: with AETHERFY_DEPLOYMENT_ID missing, a spawn id does not stand in for it.
    monkeypatch.delenv("AETHERFY_DEPLOYMENT_ID")
    monkeypatch.setenv("AETHERFY_SPAWN_ID", OWN)
    queued(transport)
    with pytest.raises(NotRunningOnAgent) as raised:
        call()
    assert raised.value.variable == "AETHERFY_DEPLOYMENT_ID"
    assert transport.calls == []


def test_without_the_supervisors_wait_a_check_still_works(transport, monkeypatch):
    monkeypatch.delenv("AETHERFY_SUPERVISOR_WAIT_URL")
    queued(transport, (200, {"waiting": False}))
    assert receive(timeout_seconds=0) is None


def _wait_ids(transport):
    return [c["body"]["wait_id"] for c in transport.calls if c["url"].endswith("/messages/receive")]


def test_a_wait_whose_request_failed_is_named_again_by_the_next_call(transport):
    # The request may have arrived and recorded the wait: a fresh id would be refused while that one is outstanding.
    queued(transport, AgentTransportError("reset"), (200, {"waiting": False}))
    with pytest.raises(AgentTransportError):
        receive()
    receive(timeout_seconds=0)
    first, second = _wait_ids(transport)
    assert first == second


def test_a_wait_whose_local_call_failed_is_named_again_by_the_next_call(transport):
    queued(transport, (200, {"waiting": True, "wait_id": "w-9"}), (503, {}), (200, {"waiting": False}))
    with pytest.raises(MessageError):
        receive()
    receive(timeout_seconds=0)
    first, second = _wait_ids(transport)
    assert first == second


def test_after_an_answer_the_next_call_names_a_new_wait(transport):
    queued(transport, (409, {"detail": {"code": "RUN_RECEIVE_ALREADY_WAITING", "message": "x"}}),
           (200, {"waiting": True, "wait_id": "w-1"}), (200, {"released": "timeout"}),
           (200, {"waiting": False}))
    with pytest.raises(MessageError):
        receive()
    assert receive(timeout_seconds=5) is None
    receive(timeout_seconds=0)
    ids = _wait_ids(transport)
    assert len(set(ids)) == 3


def test_one_receive_at_a_time_per_run():
    assert aetherfy_agent._receive_lock(OWN) is aetherfy_agent._receive_lock(OWN)
    assert aetherfy_agent._receive_lock(OWN) is not aetherfy_agent._receive_lock(PARENT)


# --- THE HISTORY, RESTARTED, PARENT ---------------------------------------------


def test_the_history_is_read_page_by_page_in_order(transport):
    queued(
        transport,
        (200, {"messages": [stored(1), stored(2)], "next_after": 2}),
        (200, {"messages": [stored(3)], "next_after": None}),
    )
    assert [m.seq for m in messages()] == [1, 2, 3]
    assert transport.calls[0]["url"] == "{0}/deployments/{1}/messages?after=0".format(API, OWN)
    assert transport.calls[1]["url"].endswith("/messages?after=2")


@pytest.mark.parametrize("attempt, expected", [(1, False), (2, True), (None, False)])
def test_restarted_reads_the_runs_attempt(transport, attempt, expected):
    run = {"id": OWN, "agent_id": "a", "state": "active"}
    if attempt is not None:
        run["attempt"] = attempt
    queued(transport, (200, run))
    assert restarted() is expected
    assert transport.calls[0]["url"] == "{0}/deployments/{1}".format(API, OWN)


def test_parent_is_the_run_that_started_this_one(transport):
    queued(transport, (200, {"id": OWN, "state": "active", "spawned_by_deployment_id": PARENT}),
           (200, {"id": OWN, "state": "active", "spawned_by_deployment_id": None}))
    assert parent() == PARENT
    assert parent() is None


# --- A SERVICE RUN --------------------------------------------------------------


def test_a_service_run_handle_sends_and_receives_with_the_runs_key_and_id(transport):
    handle = service_run({"X-Aetherfy-Run": CHILD, "X-Aetherfy-Run-Key": "afy_test_service_run_key"})
    queued(
        transport,
        (201, {"message": {"id": "x", "seq": 1}, "recipient": {"id": PARENT}}),
        (200, {"message": stored(1)}),
    )
    handle.send(PARENT, "hi")
    handle.receive()
    assert [c["api_key"] for c in transport.calls] == ["afy_test_service_run_key"] * 2
    assert transport.calls[1]["url"] == "{0}/deployments/{1}/messages/receive".format(API, CHILD)


def test_a_wait_names_the_machine_its_supervisor_set(transport, monkeypatch):
    # A service release's wait is held by the machine that runs the handler: the control plane is told which.
    monkeypatch.setenv("AETHERFY_INSTANCE_ID", "8f1c2d3e-0000-4000-8000-00000000abcd")
    queued(transport, (200, {"waiting": False}))
    receive(timeout_seconds=0)
    assert transport.calls[0]["body"].get("instance_id") == "8f1c2d3e-0000-4000-8000-00000000abcd"


def test_without_it_no_machine_is_named(transport, monkeypatch):
    monkeypatch.delenv("AETHERFY_INSTANCE_ID", raising=False)
    queued(transport, (200, {"waiting": False}))
    receive(timeout_seconds=0)
    assert "instance_id" not in transport.calls[0]["body"]
