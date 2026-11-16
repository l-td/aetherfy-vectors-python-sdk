"""
Shared by the agent tests that need to see what a call SENDS.
"""

import pytest

import aetherfy_agent


class Transport:
    """Records every request the helper makes and replies with what was queued.

    Replaces `_http.request_json` — the one function every control-plane call
    goes through — so what a test reads is exactly what would have gone out.
    """

    def __init__(self):
        self.calls = []
        self.retry_after = None
        self.reply = (200, {})
        # A test that drives several routes sets a function of (method, url)
        # returning (status, body), instead of queueing one reply.
        self.route = None

    def __call__(self, method, url, *, api_key, ua, body=None, **kwargs):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "api_key": api_key,
                "ua": ua,
                "body": body,
                "kwargs": kwargs,
            }
        )
        status, reply_body = (
            self.reply if self.route is None else self.route(method, url)
        )
        return status, reply_body, self.retry_after


@pytest.fixture
def transport(monkeypatch):
    fake = Transport()
    monkeypatch.setattr(aetherfy_agent._http, "request_json", fake)
    return fake
