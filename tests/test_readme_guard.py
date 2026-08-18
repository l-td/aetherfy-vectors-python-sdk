"""Guards README.md against the live source of THIS package.

Why this exists: until the 2026-08-18 hygiene sweep, nothing had ever read this
file. The headline migration sample called
`client.search(collection=..., vector=...)` — search() takes `collection_name`
and `query_vector` and has no **kwargs sink, so the first thing a reader copied
raised TypeError. A wrong type name and a wrong default shipped beside it.

The README is the PyPI page the moment this package is published, so it is a
released artifact with no test. It has one now.

DESIGN, and why it is not the docs-site guard:
  * docs-site's code-block guard runs in dashboard CI, which never checks out
    this repository. It validates against a COMMITTED SNAPSHOT of the SDK API
    for exactly that reason. Here the source is in the same tree, so the guard
    reads the live objects — no snapshot to go stale.
  * Static analysis only. Samples are parsed with `ast` and checked with
    `inspect`; nothing is executed and nothing touches the network. A sample
    that would hit the API is still checked for whether it could be TYPED.

SCOPE, stated honestly (same spirit as the CLI's suggested-commands guard):
  * Receivers are tracked by assignment — `client = AetherfyVectorsClient(...)`,
    `ns = memory.namespace(...)`, `with ... as client:` — and bindings persist
    across fences in document order, the way a reader following the page
    top-down accumulates them. A call on an untracked name is skipped, not
    guessed at.
  * Rebinding to a foreign class (the qdrant-client "before" samples) unbinds
    the name, so we never check qdrant methods against our own classes.
  * Keyword names are checked against inspect.signature; a method with a
    **kwargs sink accepts anything, so its keywords are not checked.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

# Packages this guard is willing to resolve. An import from anywhere else
# (qdrant_client, os) is a sample's own business.
OUR_PACKAGES = ("aetherfy_vectors", "aetherfy_memory")

FENCE_RE = re.compile(r"^```(\w+)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def read_readme() -> str:
    return README.read_text(encoding="utf-8")


def python_fences(markdown: str) -> List[str]:
    """Every ```python block, dedented enough to parse.

    Blocks nested in a numbered list are indented by markdown; ast cannot parse
    a leading indent, so strip the common prefix.
    """
    out = []
    for lang, body in FENCE_RE.findall(markdown):
        if lang != "python":
            continue
        lines = body.split("\n")
        indents = [
            len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()
        ]
        strip = min(indents) if indents else 0
        out.append("\n".join(ln[strip:] if len(ln) >= strip else ln for ln in lines))
    return out


def resolve_symbol(module_name: str, symbol: str):
    """The object a `from <module> import <symbol>` would bind, or None."""
    module = importlib.import_module(module_name)
    return getattr(module, symbol, None)


def _receiver_class(node: ast.AST, bindings: Dict[str, type]) -> Optional[type]:
    """The class bound to `x` in `x.method(...)`, if we are tracking x."""
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    return None


def _class_from_call(call: ast.Call, bindings: Dict[str, type]):
    """What a call evaluates to, when we can tell.

    Two shapes matter: constructing one of our classes, and the MemoryClient
    factories that hand back a Namespace or a Thread.
    """
    func = call.func
    if isinstance(func, ast.Name):
        for pkg in OUR_PACKAGES:
            obj = resolve_symbol(pkg, func.id)
            if inspect.isclass(obj):
                return obj
        return None  # QdrantClient and friends: unknown, so unbind
    if isinstance(func, ast.Attribute):
        owner = _receiver_class(func.value, bindings)
        if owner is None:
            return None
        method = getattr(owner, func.attr, None)
        if method is None:
            return None
        ann = getattr(method, "__annotations__", {}).get("return")
        if isinstance(ann, str):
            for pkg in OUR_PACKAGES:
                obj = resolve_symbol(pkg, ann)
                if inspect.isclass(obj):
                    return obj
        elif inspect.isclass(ann):
            return ann
    return None


def check_sample(code: str, bindings: Dict[str, type]) -> List[str]:
    """Type-level truth of one sample. Returns human-readable problems.

    `bindings` is mutated: names bound here are visible to later samples.
    """
    problems: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:  # a sample that cannot parse cannot be copied
        return [f"sample does not parse: {exc.msg} (line {exc.lineno})"]

    for node in ast.walk(tree):
        # ---- imports: every symbol must exist in the package it claims ----
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root not in OUR_PACKAGES:
                continue
            for alias in node.names:
                if resolve_symbol(node.module, alias.name) is None:
                    problems.append(
                        f"`from {node.module} import {alias.name}` — "
                        f"{alias.name} is not exported by {node.module}"
                    )

        # ---- assignments: track what our receivers are ----
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            cls = _class_from_call(node.value, bindings)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if cls is None:
                        bindings.pop(target.id, None)  # rebound to something foreign
                    else:
                        bindings[target.id] = cls
        if isinstance(node, ast.With):
            for item in node.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and item.optional_vars is not None
                    and isinstance(item.optional_vars, ast.Name)
                ):
                    cls = _class_from_call(item.context_expr, bindings)
                    if cls is not None:
                        bindings[item.optional_vars.id] = cls

        # ---- calls: method must exist, keywords must be real parameters ----
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = _receiver_class(node.func.value, bindings)
            if owner is None:
                continue
            name = node.func.attr
            method = getattr(owner, name, None)
            if method is None:
                problems.append(
                    f"`{ast.unparse(node.func.value)}.{name}(...)` — "
                    f"{owner.__name__} has no attribute {name}"
                )
                continue
            problems.extend(_check_keywords(node, owner, name, method))

        # ---- constructor keywords ----
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            cls = None
            for pkg in OUR_PACKAGES:
                obj = resolve_symbol(pkg, node.func.id)
                if inspect.isclass(obj):
                    cls = obj
                    break
            if cls is not None:
                problems.extend(_check_keywords(node, cls, "__init__", cls))

    return problems


def _check_keywords(call: ast.Call, owner, name: str, target) -> List[str]:
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return []
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []  # a **kwargs sink accepts anything
    bad = []
    for kw in call.keywords:
        if kw.arg is None:  # **spread
            continue
        if kw.arg not in params:
            allowed = ", ".join(k for k in params if k != "self")
            bad.append(
                f"`{owner.__name__}.{name}({kw.arg}=...)` — no such parameter. "
                f"Accepted: {allowed}"
            )
    return bad


# ---------------------------------------------------------------------------
# The samples themselves
# ---------------------------------------------------------------------------

def test_readme_code_samples_match_the_live_api():
    samples = python_fences(read_readme())

    # Anti-no-op: an extractor that stops matching must be RED, not silent.
    assert len(samples) >= 20, (
        f"only {len(samples)} python fences extracted from README.md — the "
        f"extractor is blind, not satisfied"
    )

    bindings: Dict[str, type] = {}
    problems: List[str] = []
    for i, sample in enumerate(samples, 1):
        for problem in check_sample(sample, bindings):
            problems.append(f"README.md python fence #{i}: {problem}")

    assert not problems, "README samples contradict the shipped API:\n  " + "\n  ".join(problems)

    # The scan is only meaningful if it actually resolved receivers.
    assert "client" in bindings, (
        "no `client` receiver was ever bound — the assignment tracking is dead "
        "and the method/keyword assertions above checked nothing"
    )


def test_the_sample_checker_actually_fires():
    """Negative control. A zero above is worthless unless the detector works.

    These are the exact defects the sweep found, plus a phantom import.
    """
    cases = [
        (
            "from aetherfy_vectors import AetherfyVectorsClient\n"
            "client = AetherfyVectorsClient(api_key='k')\n"
            "client.search(collection='c', vector=[0.1])\n",
            "no such parameter",
        ),
        (
            "from aetherfy_vectors import NoSuchSymbol\n",
            "is not exported",
        ),
        (
            "from aetherfy_vectors import AetherfyVectorsClient\n"
            "client = AetherfyVectorsClient(api_key='k')\n"
            "client.no_such_method()\n",
            "has no attribute",
        ),
        (
            "from aetherfy_vectors import AetherfyVectorsClient\n"
            "client = AetherfyVectorsClient(api_key='k'\n",
            "does not parse",
        ),
    ]
    for code, expected in cases:
        found = check_sample(code, {})
        assert any(expected in p for p in found), (
            f"checker missed {expected!r} in:\n{code}\ngot: {found}"
        )

    # ...and does NOT fire on the corrected form, or it would be useless noise.
    ok = check_sample(
        "from aetherfy_vectors import AetherfyVectorsClient\n"
        "client = AetherfyVectorsClient(api_key='k')\n"
        "client.search(collection_name='c', query_vector=[0.1], limit=5)\n",
        {},
    )
    assert ok == [], f"checker flagged a correct sample: {ok}"


def test_foreign_receivers_are_not_checked_against_our_classes():
    """The qdrant "before" samples must not be validated against our API."""
    problems = check_sample(
        "from qdrant_client import QdrantClient\n"
        "client = QdrantClient(host='localhost', port=6333)\n"
        "client.some_qdrant_only_method(foo=1)\n",
        {},
    )
    assert problems == [], f"foreign client was checked against our classes: {problems}"


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

GITHUB_OWNER = "l-td"

# Not "this repo only": cross-linking a sibling SDK is legitimate, and a rule
# that banned it would fail the day someone adds one (the JS README already
# links here). The rule that mattered is that the `aetherfy` org does not
# exist — so every URL must name a repository under an account we own.
OWNED_REPOS = {
    f"{GITHUB_OWNER}/aetherfy-vectors-python-sdk",
    f"{GITHUB_OWNER}/aetherfy-vectors-js-sdk",
    f"{GITHUB_OWNER}/aetherfy-cli",
}

# Routes that exist on docs.aetherfy.com. Liveness is not CI-checkable (no
# network in tests); the docs verification pass owns whether they resolve. This
# pins the STRINGS so a typo or an invented path cannot ship.
KNOWN_DOCS_ROUTES = {
    "https://docs.aetherfy.com",
    "https://docs.aetherfy.com/vectors",
    "https://docs.aetherfy.com/vectors/api",
    "https://docs.aetherfy.com/vectors/sdk",
    "https://docs.aetherfy.com/vectors/filtering",
    "https://docs.aetherfy.com/vectors/limits",
    "https://docs.aetherfy.com/vectors/errors",
    "https://docs.aetherfy.com/vectors/search-tuning",
    "https://docs.aetherfy.com/quickstart",
}

GITHUB_URL_RE = re.compile(r"https://github\.com/[A-Za-z0-9_.\-/]+")
DOCS_URL_RE = re.compile(r"https://docs\.aetherfy\.com[A-Za-z0-9_.\-/]*")


def test_github_urls_name_repositories_we_own():
    surfaces = {
        "README.md": read_readme(),
        "setup.py": (REPO_ROOT / "setup.py").read_text(encoding="utf-8"),
    }
    found = 0
    bad = []
    for where, text in surfaces.items():
        for url in GITHUB_URL_RE.findall(text):
            found += 1
            # A clone URL carries a `.git` suffix; same repository.
            repo = "/".join(
                url.replace("https://github.com/", "").split("/")[:2]
            ).removesuffix(".git")
            if repo not in OWNED_REPOS:
                bad.append(
                    f"{where} links {url} — {repo} is not a repository we own. "
                    f"Owned: {', '.join(sorted(OWNED_REPOS))}. "
                    f"github.com/aetherfy/* is an org that does not exist."
                )
    assert not bad, "\n  ".join(bad)
    assert found >= 3, (
        f"only {found} github.com URLs found across README/setup.py — the URL "
        f"scan is dead and the assertion above is vacuous"
    )


def test_docs_links_are_known_routes():
    found = 0
    for url in DOCS_URL_RE.findall(read_readme()):
        found += 1
        assert url.rstrip("/") in KNOWN_DOCS_ROUTES, (
            f"README links {url}, which is not a known docs route. Add it to "
            f"KNOWN_DOCS_ROUTES only after confirming it resolves on the live "
            f"site — docs.aetherfy.com/api and vectors.aetherfy.com/docs were "
            f"both invented and both 404'd."
        )
    assert found >= 2, f"only {found} docs URLs found — the scan is dead"


# ---------------------------------------------------------------------------
# Performance claims
# ---------------------------------------------------------------------------

# Owner ruling (2026-08-18): performance claims stay QUALITATIVE. A number with
# a performance unit is a promise nothing in this repo measures.
PERF_CLAIM_PATTERNS = [
    (re.compile(r"sub-\s*\d+\s*ms", re.I), "a 'sub-Nms' latency promise"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*ms\b", re.I), "a latency figure in ms"),
    (re.compile(r"\b[\d,]+\+?\s*(?:QPS|queries per second|requests per second)", re.I),
     "a throughput figure"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\+?\s*(?:cache|hit rate|uptime|availability|SLA)", re.I),
     "a cache/uptime percentage"),
    (re.compile(r"(?:cache hit rate|uptime|availability|SLA)[^.\n]{0,24}?\b\d+(?:\.\d+)?\s*%", re.I),
     "a cache/uptime percentage"),
]


def prose_only(markdown: str) -> str:
    """The README minus fenced code — claims are prose; `avg_latency_ms` is not."""
    return FENCE_RE.sub("", markdown)


def test_no_numeric_performance_claims_in_prose():
    prose = prose_only(read_readme())
    hits = []
    for line_no, line in enumerate(prose.split("\n"), 1):
        for pattern, why in PERF_CLAIM_PATTERNS:
            m = pattern.search(line)
            if m:
                hits.append(f"line {line_no}: {m.group(0)!r} — {why}")
    assert not hits, (
        "README states numeric performance claims. The owner ruled these stay "
        "qualitative: the measured numbers live in the benchmark harness, at "
        "values these did not match, and a figure in a registry README is a "
        "promise a reader can hold us to.\n  " + "\n  ".join(hits)
    )


def test_perf_claim_patterns_actually_fire():
    """Negative control for the tripwire, including what must NOT trip it."""
    for bad in [
        "sub-50ms latency worldwide",
        "Average latency: 12 ms",
        "100,000+ queries per second",
        "94%+ cache hit rate",
        "Availability: 99.9% SLA",
    ]:
        assert any(p.search(bad) for p, _ in PERF_CLAIM_PATTERNS), f"missed: {bad}"

    for fine in [
        "100% compatible with `qdrant-client` API",           # compatibility, not perf
        "up to **512 points** in one round trip",
        "1000 points/call, 10 MB/response",
        "Python 3.9, 3.10, 3.11, 3.12",
        "VectorConfig(size=128, distance=DistanceMetric.COSINE)",
    ]:
        assert not any(p.search(fine) for p, _ in PERF_CLAIM_PATTERNS), (
            f"false positive on: {fine}"
        )


def test_install_line_is_left_alone():
    """The registry is an owner gate, not an error. Do not hedge pip install."""
    assert "pip install aetherfy-vectors" in read_readme(), (
        "the install line must keep the real package name — an unpublished "
        "registry is an owner gate, not something to annotate around"
    )
