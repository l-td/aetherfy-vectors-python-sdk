"""The qdrant-client compatibility contract and the end of the **kwargs sinks.

Until 1.2.0 the constructor and most methods ended in ``**kwargs`` that
nothing read. ``region=`` was renamed to ``api_region=`` on 2026-06-29, and a
caller still passing ``region="eu-central-1"`` got no error, no warning, and a
client routed to the default endpoint. The e2e region tests were written
against the old name and could never have passed, and nothing said so.

What is pinned here:
  * the constructor has no sink: an unknown argument is a TypeError;
  * every method shared with qdrant-client accepts exactly its own parameters
    plus the qdrant-client parameters classified in
    ``aetherfy_vectors.qdrant_compat``, with each classification's behaviour;
  * the classification is DERIVED against the real qdrant-client signatures at
    the pinned version, so a qdrant parameter that is neither ours nor
    classified fails here instead of drifting silently.

qdrant-client is a dev dependency (requirements-dev.txt). It is imported
unconditionally: a missing install must fail this file, never skip it, or
the derivation would read as green while checking nothing.
"""

import importlib.metadata
import inspect
import re
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.exceptions import ValidationError
from aetherfy_vectors.qdrant_compat import (
    IGNORE,
    QDRANT_CLIENT_VERSION,
    QDRANT_COMPAT,
    REFUSE,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _named(func) -> Dict[str, inspect.Parameter]:
    """Named parameters of a callable: no self, no *args, no **kwargs."""
    return {
        name: p
        for name, p in inspect.signature(func).parameters.items()
        if name != "self"
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def _has_sink(func) -> bool:
    return any(
        p.kind is p.VAR_KEYWORD for p in inspect.signature(func).parameters.values()
    )


def _shared_methods():
    """Public methods of ours that qdrant-client also has, enumerated from
    both classes rather than listed, so a method added to either side is
    picked up without editing this file."""
    ours = {
        name
        for name, value in vars(AetherfyVectorsClient).items()
        if callable(value) and not name.startswith("_")
    }
    return sorted(n for n in ours if callable(getattr(QdrantClient, n, None)))


SHARED = _shared_methods()


# ---------------------------------------------------------------------------
# §1 the constructor
# ---------------------------------------------------------------------------


class TestConstructorHasNoSink:
    def test_the_pre_rename_region_name_raises(self, api_key, monkeypatch):
        # THE defect in one assertion: on 1.1.0 this constructed a client
        # routed to the default endpoint and said nothing.
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        monkeypatch.delenv("AETHERFY_VECTORS_API_REGION", raising=False)
        with pytest.raises(TypeError, match="unexpected keyword argument 'region'"):
            AetherfyVectorsClient(api_key=api_key, region="iad")

    def test_a_misspelt_argument_raises(self, api_key):
        with pytest.raises(
            TypeError, match="unexpected keyword argument 'api_regoin'"
        ):
            AetherfyVectorsClient(api_key=api_key, api_regoin="eu-central-1")

    def test_a_qdrant_constructor_argument_raises(self, api_key):
        # The migration replaces qdrant's constructor call wholesale; none of
        # its arguments is part of the promise.
        with pytest.raises(TypeError, match="unexpected keyword argument 'host'"):
            AetherfyVectorsClient(api_key=api_key, host="localhost")

    def test_the_constructor_signature_has_no_var_keyword(self):
        assert not _has_sink(AetherfyVectorsClient.__init__)

    def test_an_invalid_api_region_still_raises_value_error(self, api_key, monkeypatch):
        # The validation the e2e tests expected was always there; the sink is
        # why `region=` never reached it.
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        with pytest.raises(ValueError, match="api_region must be one of"):
            AetherfyVectorsClient(api_key=api_key, api_region="iad")

    def test_a_valid_api_region_still_resolves(self, api_key, monkeypatch):
        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        resp = Mock()
        resp.status_code = 200
        resp.content = (
            b'{"us-east-1": "https://vectors-iad.aetherfy.run",'
            b' "eu-central-1": "https://vectors-fra.aetherfy.run"}'
        )
        with patch("aetherfy_vectors.client.requests.get", return_value=resp):
            c = AetherfyVectorsClient(api_key=api_key, api_region="eu-central-1")
        assert c.api_region == "eu-central-1"
        assert c.endpoint == "https://vectors-fra.aetherfy.run"


# ---------------------------------------------------------------------------
# §4.4 the classification is derived against the real qdrant-client
# ---------------------------------------------------------------------------


class TestClassificationIsDerived:
    def test_the_installed_qdrant_client_is_the_pinned_one(self):
        installed = importlib.metadata.version("qdrant-client")
        assert installed == QDRANT_CLIENT_VERSION, (
            f"qdrant-client {installed} is installed but the contract was "
            f"derived from {QDRANT_CLIENT_VERSION}. Install the pin from "
            "requirements-dev.txt, or, to upgrade, change the pin in BOTH "
            "places and classify what the new version added."
        )

    def test_requirements_dev_pins_the_same_version(self):
        text = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        pins = re.findall(r"^qdrant-client==(\S+)", text, re.MULTILINE)
        assert pins == [QDRANT_CLIENT_VERSION], (
            f"requirements-dev.txt pins qdrant-client {pins}; the contract was "
            f"derived from {QDRANT_CLIENT_VERSION}"
        )

    def test_the_enumeration_found_the_methods_it_must(self):
        # The derivation below iterates SHARED. If the enumeration broke and
        # found nothing, every per-method check would pass over an empty list.
        for method in ("upsert", "search", "scroll", "retrieve", "create_collection"):
            assert method in SHARED, f"{method} missing from {SHARED}"

    @pytest.mark.parametrize("method", SHARED)
    def test_every_qdrant_parameter_is_ours_or_classified(self, method):
        qdrant = set(_named(getattr(QdrantClient, method)))
        ours = set(_named(getattr(AetherfyVectorsClient, method)))
        classified = set(QDRANT_COMPAT.get(method, {}))

        unclassified = qdrant - ours - classified
        assert not unclassified, (
            f"qdrant-client {QDRANT_CLIENT_VERSION} {method}() has "
            f"{sorted(unclassified)}, which {method}() neither names nor "
            "classifies. Add each to QDRANT_COMPAT with a reason (IGNORE only "
            "if our behaviour already gives the caller what they asked for), "
            "or implement it as a named parameter."
        )
        stale = classified - qdrant
        assert not stale, (
            f"QDRANT_COMPAT[{method!r}] classifies {sorted(stale)}, which "
            f"qdrant-client {QDRANT_CLIENT_VERSION} {method}() does not have"
        )
        double = classified & ours
        assert not double, (
            f"{sorted(double)} is both a named parameter of {method}() and "
            "classified; a named parameter is honoured, so drop the entry"
        )

    @pytest.mark.parametrize("method", SHARED)
    def test_our_positional_parameters_are_a_prefix_of_qdrants(self, method):
        # A qdrant-client call passing arguments by position must bind them to
        # the parameters qdrant-client means, or fail. Where our order
        # diverged (search's third positional was `limit`, qdrant's is
        # `query_filter`) a positional call bound the wrong value silently.
        # Everything past the shared prefix is keyword-only, so a longer
        # positional call is a TypeError.
        def positional(func):
            return [
                name
                for name, p in _named(func).items()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]

        ours = positional(getattr(AetherfyVectorsClient, method))
        qdrant = positional(getattr(QdrantClient, method))
        assert ours == qdrant[: len(ours)], (
            f"{method}(): our positional parameters {ours} are not a prefix of "
            f"qdrant-client {QDRANT_CLIENT_VERSION}'s {qdrant}. Make every "
            "parameter after the shared prefix keyword-only (*)."
        )

    @pytest.mark.parametrize("method", SHARED)
    def test_a_sink_exists_exactly_where_something_is_classified(self, method):
        has_sink = _has_sink(getattr(AetherfyVectorsClient, method))
        assert has_sink == bool(QDRANT_COMPAT.get(method)), (
            f"{method}(): **kwargs present={has_sink} but classified "
            f"entries={sorted(QDRANT_COMPAT.get(method, {}))}. A method with "
            "nothing classified must not take **kwargs at all."
        )

    def test_the_table_names_only_shared_methods(self):
        assert set(QDRANT_COMPAT) <= set(SHARED), sorted(set(QDRANT_COMPAT) - set(SHARED))

    def test_qdrant_default_values_are_accepted_by_every_refusal(self):
        # Passing qdrant's own default says nothing beyond not passing it.
        for method, rules in QDRANT_COMPAT.items():
            params = _named(getattr(QdrantClient, method))
            for name, rule in rules.items():
                if rule.kind != REFUSE:
                    continue
                default = params[name].default
                assert default in rule.accepted, (
                    f"{method}({name}={default!r}) is qdrant-client's default "
                    "but the refusal does not accept it"
                )


# ---------------------------------------------------------------------------
# §4.3 behaviour, per method
# ---------------------------------------------------------------------------

# The contract, pinned independently of the table it checks: a REFUSE turned
# into an IGNORE in qdrant_compat must fail here, not be re-read from there.
EXPECTED: Dict[str, Dict[str, str]] = {
    "close": {"grpc_grace": IGNORE},
    "count": {"shard_key_selector": REFUSE},
    "create_collection": {
        name: REFUSE
        for name in (
            "sparse_vectors_config",
            "shard_number",
            "sharding_method",
            "replication_factor",
            "write_consistency_factor",
            "on_disk_payload",
            "hnsw_config",
            "optimizers_config",
            "wal_config",
            "quantization_config",
            "strict_mode_config",
            "init_from",
        )
    },
    **{
        m: {"wait": IGNORE, "ordering": REFUSE, "shard_key_selector": REFUSE}
        for m in ("delete", "delete_payload", "overwrite_payload", "set_payload", "upsert")
    },
    "retrieve": {"consistency": REFUSE, "shard_key_selector": REFUSE},
    "scroll": {"consistency": REFUSE, "shard_key_selector": REFUSE},
    "search": {
        "append_payload": REFUSE,
        "consistency": REFUSE,
        "shard_key_selector": REFUSE,
    },
}

# A value each refusal must reject (anything not describing our behaviour).
REFUSED_VALUE = {"ordering": "strong", "append_payload": False}
# Values that describe our behaviour, so the refusal must let them through.
ACCEPTED_VALUES = {
    "ordering": [None, "weak", qm.WriteOrdering.WEAK],
    "append_payload": [True],
}

POINT = {"id": 1, "vector": [0.1, 0.2], "payload": {"k": "v"}}

# Minimal positional arguments to reach each method's body.
CALL_ARGS: Dict[str, Tuple[Any, ...]] = {
    "close": (),
    "collection_exists": ("c",),
    "count": ("c",),
    "create_collection": ("c", {"size": 2, "distance": "Cosine"}),
    "delete": ("c", [1]),
    "delete_collection": ("c",),
    "delete_payload": ("c", ["k"], [1]),
    "get_collection": ("c",),
    "get_collections": (),
    "overwrite_payload": ("c", {"k": "v"}, [1]),
    "retrieve": ("c", [1]),
    "scroll": ("c",),
    "search": ("c", [0.1, 0.2]),
    "set_payload": ("c", {"k": "v"}, [1]),
    "upsert": ("c", [POINT]),
}


def _response(payload):
    resp = Mock()
    resp.status_code = 200
    resp.content = b"{}"
    resp.json.return_value = payload
    return resp


@pytest.fixture
def wired(client, mock_requests):
    """A client whose HTTP answers every call with a plausible 200, and whose
    caches already know collection "c" so upsert makes no schema reads."""

    def answer(*args, **kwargs):
        url = kwargs.get("url", "")
        if url.endswith("/points/search"):
            return _response({"result": []})
        if url.endswith("/points/retrieve"):
            return _response({"result": []})
        return _response(
            {
                "result": {"points": [], "next_page_offset": None, "count": 0},
                "collections": [],
            }
        )

    mock_requests.request.side_effect = answer
    client._schema_cache["c"] = {"size": 2, "distance": "Cosine", "etag": None}
    client._payload_schema_cache["c"] = {
        "schema": None,
        "enforcement_mode": "off",
        "etag": None,
    }
    return client, mock_requests


def _call(client, method, **kwargs):
    return getattr(client, method)(*CALL_ARGS[method], **kwargs)


def _bodies(mock_requests):
    return [c.kwargs.get("json") for c in mock_requests.request.call_args_list]


def _all_keys(obj):
    """Every dict key anywhere in a request body."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


class TestPerMethodBehaviour:
    def test_the_table_is_the_expected_contract(self):
        actual = {m: {n: r.kind for n, r in rules.items()} for m, rules in QDRANT_COMPAT.items()}
        assert actual == EXPECTED

    def test_every_shared_method_has_a_call_fixture(self):
        # Otherwise the parametrised checks below would silently skip one.
        assert sorted(CALL_ARGS) == SHARED

    @pytest.mark.parametrize("method", SHARED)
    def test_a_name_qdrant_does_not_have_raises(self, wired, method):
        client, mock_requests = wired
        with pytest.raises(
            TypeError, match="unexpected keyword argument 'not_a_qdrant_arg'"
        ):
            _call(client, method, not_a_qdrant_arg=1)
        assert mock_requests.request.call_count == 0, "raised after sending"

    @pytest.mark.parametrize(
        "method,name",
        [(m, n) for m, rules in EXPECTED.items() for n, k in rules.items() if k == REFUSE],
    )
    def test_a_refused_argument_raises_naming_itself(self, wired, method, name):
        client, mock_requests = wired
        value = REFUSED_VALUE.get(name, {"some": "setting"})
        with pytest.raises(TypeError) as exc:
            _call(client, method, **{name: value})
        message = str(exc.value)
        # The name, AND the refusal wording: "unexpected keyword argument
        # 'x'" also contains the name, and would mean the entry was lost from
        # the table rather than refused by it.
        assert f"'{name}'" in message
        assert "does not support qdrant-client's" in message, message
        assert mock_requests.request.call_count == 0, "raised after sending"

    @pytest.mark.parametrize(
        "method,name",
        [(m, n) for m, rules in EXPECTED.items() for n, k in rules.items() if k == REFUSE],
    )
    def test_a_refused_argument_passed_as_our_behaviour_is_accepted(
        self, wired, method, name
    ):
        client, mock_requests = wired
        for value in ACCEPTED_VALUES.get(name, [None]):
            mock_requests.request.reset_mock()
            _call(client, method, **{name: value})
            if method != "close":
                assert mock_requests.request.call_count >= 1
            for body in _bodies(mock_requests):
                assert name not in set(_all_keys(body)), (
                    f"{method}({name}=...) leaked into the request body: {body}"
                )

    @pytest.mark.parametrize(
        "method,name",
        [(m, n) for m, rules in EXPECTED.items() for n, k in rules.items() if k == IGNORE],
    )
    def test_an_ignore_safe_argument_is_accepted_and_not_sent(
        self, wired, method, name
    ):
        client, mock_requests = wired
        values = [True, False] if name == "wait" else [5]
        for value in values:
            mock_requests.request.reset_mock()
            _call(client, method, **{name: value})
            if method != "close":
                assert mock_requests.request.call_count >= 1
            for body in _bodies(mock_requests):
                assert name not in set(_all_keys(body)), body


def _readme_contract():
    """The README's Migration Compatibility section: its method list, and its
    table expanded to (method, argument, behaviour) triples."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("### Migration Compatibility")
    section = text[start : text.index("\n## ", start)]
    listed = re.search(r"keyword arguments:\n(.*?)\n\n", section, re.DOTALL)
    assert listed, "the method list paragraph is gone"
    methods = re.findall(r"`(\w+)`", listed.group(1))
    triples = set()
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] in ("Method",) or set(cells[0]) <= set("-"):
            continue
        here = cells[2]
        if here.startswith("accepted, no effect"):
            kind = IGNORE
        elif here.startswith("refused"):
            kind = REFUSE
        elif here == "honoured":
            kind = "honour"
        else:
            raise AssertionError(f"unrecognised behaviour {here!r} in: {line}")
        for m in re.findall(r"`(\w+)`", cells[0]):
            for a in re.findall(r"`(\w+)`", cells[1]):
                triples.add((m, a, kind))
    return methods, triples


class TestReadmeStatesTheContract:
    def test_the_method_list_is_the_shared_methods(self):
        methods, _ = _readme_contract()
        assert sorted(methods) == SHARED

    def test_the_table_matches_the_classification(self):
        _, triples = _readme_contract()
        documented = {t for t in triples if t[2] != "honour"}
        actual = {
            (m, a, rule.kind) for m, rules in QDRANT_COMPAT.items() for a, rule in rules.items()
        }
        assert documented == actual, {
            "in README only": sorted(documented - actual),
            "in QDRANT_COMPAT only": sorted(actual - documented),
        }

    def test_every_honoured_row_is_a_real_shared_parameter(self):
        _, triples = _readme_contract()
        honoured = [(m, a) for m, a, k in triples if k == "honour"]
        assert honoured, "the honoured rows are gone"
        for m, a in honoured:
            assert a in _named(getattr(AetherfyVectorsClient, m)), (m, a)
            assert a in _named(getattr(QdrantClient, m)), (m, a)


class TestHonouredQdrantParameters:
    @pytest.mark.parametrize(
        "method", ["create_collection", "delete_collection", "retrieve", "scroll", "count", "search"]
    )
    def test_timeout_bounds_the_request(self, wired, method):
        client, mock_requests = wired
        _call(client, method, timeout=7)
        timeouts = [c.kwargs["timeout"] for c in mock_requests.request.call_args_list]
        # One attempt, given what remains of the 7 s deadline: a hair under 7.
        assert len(timeouts) == 1 and 6.5 < timeouts[0] <= 7, timeouts

    @pytest.mark.parametrize(
        "method", ["create_collection", "delete_collection", "retrieve", "scroll", "count", "search"]
    )
    def test_no_timeout_keeps_the_client_policy(self, wired, method):
        client, mock_requests = wired
        _call(client, method)
        timeouts = [c.kwargs["timeout"] for c in mock_requests.request.call_args_list]
        # The client fixture's timeout is 10.0; every body here is far below
        # the body-aware threshold.
        assert timeouts == [10.0], timeouts

    def test_positional_arguments_past_the_shared_prefix_raise(self, wired):
        client, mock_requests = wired
        # qdrant-client's third positional is query_filter; ours was limit.
        with pytest.raises(TypeError):
            client.search("c", [0.1, 0.2], {"must": []})
        assert mock_requests.request.call_count == 0


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


class _AlwaysTimesOut:
    """A transport that never answers. Each attempt costs the timeout it was
    given (or `per_attempt`, if smaller, to model a fast failure) on the fake
    clock, then raises requests.Timeout."""

    def __init__(self, clock, per_attempt=None):
        self.clock = clock
        self.per_attempt = per_attempt
        self.attempts = []  # (clock time the attempt started, timeout given)

    def __call__(self, *args, **kwargs):
        import requests

        given = kwargs["timeout"]
        self.attempts.append((self.clock.now, given))
        cost = given if self.per_attempt is None else min(given, self.per_attempt)
        self.clock.now += cost
        raise requests.Timeout("fake transport: no answer")


@pytest.fixture
def clock():
    """The fake clock, installed in both modules that read time, with backoff
    jitter pinned to its maximum so the delays are exactly 1 s, 2 s, 4 s."""
    fake = _FakeClock()
    with patch("aetherfy_vectors.client.time", fake), patch(
        "aetherfy_vectors.utils.time", fake
    ), patch("aetherfy_vectors.utils.random.random", return_value=1.0):
        yield fake


class TestMethodTimeoutIsAWholeCallDeadline:
    """qdrant-client's method `timeout` bounds the operation. Ours used to
    bound each attempt and then retry up to 3 times with backoff, so
    timeout=5 could take ~18 s. It is a deadline for the whole call now."""

    def test_a_hanging_transport_is_bounded_by_the_deadline(self, wired, clock):
        from aetherfy_vectors.exceptions import RequestTimeoutError

        client, mock_requests = wired
        transport = _AlwaysTimesOut(clock)
        mock_requests.request.side_effect = transport
        with pytest.raises(RequestTimeoutError, match="deadline"):
            client.retrieve("c", [1], timeout=0.5)
        # One attempt, given the whole budget, which it used up. The 1 s
        # backoff would end past the deadline, so it was not taken.
        assert transport.attempts == [(0.0, 0.5)]
        assert clock.sleeps == []
        assert clock.now == 0.5

    def test_retries_fit_inside_the_deadline_and_none_starts_after_it(
        self, wired, clock
    ):
        from aetherfy_vectors.exceptions import RequestTimeoutError

        client, mock_requests = wired
        transport = _AlwaysTimesOut(clock, per_attempt=0.05)
        mock_requests.request.side_effect = transport
        with pytest.raises(RequestTimeoutError):
            client.retrieve("c", [1], timeout=2.5)
        # Attempt 1 at 0 s gets all 2.5 s and fails at 0.05 s. The 1 s
        # backoff ends at 1.05 s, inside the deadline, so it is taken, and
        # attempt 2 gets only the 1.45 s that remain. It fails at 1.10 s, and
        # the 2 s backoff would end at 3.10 s, past the deadline: not taken,
        # and no third attempt.
        assert transport.attempts == [
            (0.0, 2.5),
            (pytest.approx(1.05), pytest.approx(1.45)),
        ]
        assert clock.sleeps == [1.0]
        assert clock.now == pytest.approx(1.10)

    def test_without_a_method_timeout_the_client_timeout_is_per_attempt(
        self, wired, clock
    ):
        from aetherfy_vectors.exceptions import RequestTimeoutError

        client, mock_requests = wired
        transport = _AlwaysTimesOut(clock, per_attempt=0.01)
        mock_requests.request.side_effect = transport
        with pytest.raises(RequestTimeoutError):
            client.retrieve("c", [1])
        # The constructor's timeout keeps its meaning: each of the four
        # attempts (1 + 3 retries) gets the full 10.0 s, with the whole
        # 1 + 2 + 4 s backoff between them.
        assert [given for _, given in transport.attempts] == [10.0] * 4
        assert clock.sleeps == [1.0, 2.0, 4.0]


class TestScrollOrderBy:
    @pytest.mark.parametrize(
        "order_by", ["ts", {"key": "ts", "direction": "desc"}]
    )
    def test_scroll_order_by_is_sent_verbatim(self, wired, order_by):
        client, mock_requests = wired
        client.scroll("c", order_by=order_by)
        assert _bodies(mock_requests)[-1]["order_by"] == order_by

    def test_scroll_without_order_by_sends_no_order_by(self, wired):
        client, mock_requests = wired
        client.scroll("c")
        assert "order_by" not in _bodies(mock_requests)[-1]

    def test_scroll_order_by_of_another_type_is_rejected(self, wired):
        client, mock_requests = wired
        with pytest.raises(ValidationError, match="order_by must be"):
            client.scroll("c", order_by=qm.OrderBy(key="ts"))
        assert mock_requests.request.call_count == 0
