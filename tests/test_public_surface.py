"""
Every exception the SDK raises, the customer can import.

Why this exists: on 2026-09-24 the docs guard for the JavaScript SDK learned
what that package's root really exported, and found the memory client THROWING
a class the root did not export — `except`/`instanceof` on it was impossible
for a root importer. Nothing had noticed, and nothing would have. This is the
Python half of the test that makes the class structural (the JavaScript half
is `tests/unit/public-surface.test.ts` in aetherfy-vectors-js-sdk).

For every top-level package this distribution ships (setup.py's own
`find_packages(exclude=...)` call, read and applied here, never hard-coded, so
a new package is covered the day setup.py ships it): every `raise X(...)` /
`raise X` in every module of the package, parsed with the stdlib `ast` and
with X RESOLVED in the raising
module's namespace — not matched by name — must be importable from the root
of the SDK package that DEFINES it: named in that root's `__all__` (the
convention all three roots follow) and bound there to the very same class.

The DEFINING package, not the raising one: aetherfy_memory is layered on
aetherfy_vectors and raises some of its errors (PointNotFoundError from the
metadata writers of a Namespace or Thread). A memory user already imports
those from aetherfy_vectors; re-exporting every vectors error from every
package built on it would be surface for its own sake.

The raise walk cannot see a class that only a FACTORY builds —
`raise parse_error_response(...)` is decided at runtime, and every class that
function returns (CollectionNotFoundError, ConflictError, CollectionInUseError,
...) was exported only because someone remembered to. So, second property:
every exception class a package DEFINES that descends from the package's own
BASE must be exported from that package's root. Classes are found as
module-level `ClassDef`s in the package's source, their parents resolved from
the `ast` bases in the defining module's namespace. A package's base is an
exception class it defines whose SDK parent it does NOT define:
AetherfyVectorsException (parent: Exception), AetherfyMemoryException (parent:
AetherfyVectorsException, defined in aetherfy_vectors), AgentError (likewise).
That anchor is structural — the source tree, not the exports — because a
Python package is a directory and partitions the classes cleanly; dropping a
base from `__all__` cannot take its subtree out of the check.

Not checked, and why:
  - Built-in and third-party exceptions (ValueError, TypeError, RuntimeError):
    importable by definition.
  - A bare `raise name` where `name` is a local (re-raising a caught value):
    it is whatever was caught.
  - Signature types. Python has no separate type-export step: a type in a
    public signature is reachable through the signature's own module.

ANTI-NO-OP: every package must have parsed more than one module, found raise
sites of its own, found a base, and found at least one class descending from
it. A test that walked nothing looks exactly like a test that verified
everything.
"""

import ast
import builtins
import fnmatch
import importlib
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _shipped_packages() -> List[str]:
    """The top-level packages setup.py ships, from its own find_packages call."""
    setup = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(setup)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "find_packages"
    ]
    if len(calls) != 1:
        raise AssertionError(
            f"setup.py has {len(calls)} find_packages(...) calls; this test "
            "reads the one that decides what ships, so it needs exactly one."
        )
    kwargs: Dict[str, Any] = {}
    for kw in calls[0].keywords:
        if kw.arg is None:
            raise AssertionError("setup.py passes find_packages a **splat.")
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    if set(kwargs) - {"exclude"} or calls[0].args:
        raise AssertionError(
            "setup.py's find_packages takes arguments besides exclude=; teach "
            "this test what they mean before trusting its package list."
        )
    # find_packages' own rule for a top-level package — a directory holding an
    # __init__.py whose name fnmatch-es no exclude pattern — applied here rather
    # than imported: setuptools is not guaranteed in the test environment
    # (Python 3.12 ships without it and `pip install -e .` builds in isolation).
    excluded = kwargs.get("exclude", ())
    return sorted(
        path.name
        for path in REPO_ROOT.iterdir()
        if (path / "__init__.py").is_file()
        and not any(fnmatch.fnmatchcase(path.name, pat) for pat in excluded)
    )


PACKAGES = _shipped_packages()


class RaiseSite(NamedTuple):
    exception: type
    at: str


class Walk(NamedTuple):
    modules: int
    own: List[RaiseSite]
    skipped: int


def _dotted(node: ast.expr) -> Optional[Sequence[str]]:
    """`Name` / `a.b.C` as a list of names, or None for anything else."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return list(reversed(parts))


_MISSING = object()


def _resolve(namespace: Dict[str, object], dotted: Sequence[str]) -> object:
    """Look `dotted` up the way the raising module would at runtime."""
    head = namespace.get(dotted[0], getattr(builtins, dotted[0], _MISSING))
    for name in dotted[1:]:
        if head is _MISSING:
            break
        head = getattr(head, name, _MISSING)
    return head


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _walk(package: str) -> Walk:
    own: List[RaiseSite] = []
    skipped = 0
    paths = sorted((REPO_ROOT / package).rglob("*.py"))
    for path in paths:
        module = importlib.import_module(_module_name(path))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            at = f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}"
            is_call = isinstance(node.exc, ast.Call)
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            dotted = _dotted(target)
            resolved = _MISSING if dotted is None else _resolve(vars(module), dotted)
            if resolved is _MISSING:
                if is_call:
                    # Never skipped: an unresolvable `raise X(...)` is exactly
                    # the site this test could otherwise pass without having
                    # looked at.
                    raise AssertionError(f"Cannot resolve what is raised at {at}.")
                skipped += 1  # re-raising a caught value
                continue
            if not (isinstance(resolved, type) and issubclass(resolved, BaseException)):
                skipped += 1  # a factory call; what it builds is runtime's
                continue
            if resolved.__module__.split(".")[0] not in PACKAGES:
                skipped += 1  # built-in or third-party
                continue
            own.append(RaiseSite(resolved, at))
    return Walk(len(paths), own, skipped)


class Defined(NamedTuple):
    exception: type
    parent: Optional[type]  # its SDK parent, None when that is not an SDK class
    at: str


def _defined_exceptions(package: str) -> List[Defined]:
    """Every module-level exception class the package's source defines."""
    out: List[Defined] = []
    for path in sorted((REPO_ROOT / package).rglob("*.py")):
        module = importlib.import_module(_module_name(path))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            at = f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}"
            cls = getattr(module, node.name, _MISSING)
            if not isinstance(cls, type):
                raise AssertionError(f"Cannot resolve the class defined at {at}.")
            if not issubclass(cls, BaseException):
                continue
            parents = []
            for base in node.bases:
                dotted = _dotted(base)
                resolved = (
                    _MISSING if dotted is None else _resolve(vars(module), dotted)
                )
                if not isinstance(resolved, type):
                    raise AssertionError(f"Cannot resolve a base of the class at {at}.")
                parents.append(resolved)
            sdk_parent = next(
                (p for p in parents if p.__module__.split(".")[0] in PACKAGES), None
            )
            out.append(Defined(cls, sdk_parent, at))
    return out


def _bases(package: str, defined: List[Defined]) -> List[type]:
    """Defined exception classes whose SDK parent this package does not define."""
    own = {id(d.exception) for d in defined}
    return [d.exception for d in defined if d.parent is None or id(d.parent) not in own]


def _exported(root: str) -> Dict[int, str]:
    """id(object) -> name, for every name in the root's `__all__`."""
    module = importlib.import_module(root)
    names = getattr(module, "__all__", None)
    if names is None:
        raise AssertionError(f"{root} declares no __all__.")
    return {id(getattr(module, name)): name for name in names if hasattr(module, name)}


@pytest.mark.parametrize("package", PACKAGES)
def test_walk_found_work(package: str) -> None:
    walk = _walk(package)
    assert walk.modules > 1, f"{package}: parsed {walk.modules} module(s)"
    assert walk.own, f"{package}: found no raise site of an SDK exception"
    defined = _defined_exceptions(package)
    bases = _bases(package, defined)
    assert bases, f"{package}: found no base exception class"
    assert any(
        d.exception not in bases and issubclass(d.exception, tuple(bases))
        for d in defined
    ), f"{package}: found no exception class descending from {bases}"


@pytest.mark.parametrize("package", PACKAGES)
def test_every_raised_sdk_exception_is_importable(package: str) -> None:
    unexported: Dict[type, List[str]] = {}
    for site in _walk(package).own:
        home = site.exception.__module__.split(".")[0]
        if id(site.exception) not in _exported(home):
            unexported.setdefault(site.exception, []).append(site.at)
    failures = [
        f"{exc.__name__} ({exc.__module__}) is raised at {', '.join(sites)} "
        f"but is not importable from '{exc.__module__.split('.')[0]}' "
        "(not bound to that class under any name in its __all__)"
        for exc, sites in sorted(unexported.items(), key=lambda i: i[0].__name__)
    ]
    assert not failures, "\n" + "\n".join(failures)


def test_packages_are_the_shipped_ones() -> None:
    # A find_packages that found nothing would make every parametrized test
    # above vanish rather than fail.
    assert {"aetherfy_vectors", "aetherfy_memory", "aetherfy_agent"} <= set(
        PACKAGES
    ), PACKAGES


@pytest.mark.parametrize("package", PACKAGES)
def test_every_exception_in_its_hierarchy_is_importable(package: str) -> None:
    defined = _defined_exceptions(package)
    bases = tuple(_bases(package, defined))
    exported = _exported(package)
    failures = [
        f"{d.exception.__name__} ({d.at}) "
        + (
            "is this package's base exception"
            if d.exception in bases
            else "descends from "
            + next(b.__name__ for b in bases if issubclass(d.exception, b))
        )
        + f" but is not importable from '{package}' (not in its __all__)"
        for d in defined
        if issubclass(d.exception, bases) and id(d.exception) not in exported
    ]
    assert not failures, "\n" + "\n".join(failures)
