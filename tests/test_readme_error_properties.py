"""Guards the README's error-object promises against the classes that raise.

WHY THIS EXISTS: the README tells integrators what a caught exception carries —
`e.retry_after`, `e.quota_type`, `e.errors`. Nothing tied those claims to the
classes. The README could document a property no constructor ever assigns and
every suite in this repository would stay green while the reader's handler
printed `None`. Publication makes the README the PyPI page, so those claims
become contractual; this is the check that says they are true.

It is the twin of tests/test_readme_guard.py, which checks the SAMPLES against
the API. That guard checks calls; this one checks the SHAPE OF WHAT IS CAUGHT.

THE CHECK, narrow on purpose, and its edges stated honestly:

  * A CLAIM is `except <OurError> as e:` inside a README ```python fence, plus
    an `e.<prop>` read anywhere in that handler's body. That is the ONLY claim
    form parsed. Prose is deliberately not parsed: prose is ambiguous about
    whose property it names — the Limits section's "a structured `error.code`"
    is the SERVER envelope's key, not an attribute of any class in this
    package, and a parser that read it as one would red on a true sentence.

  * An ASSIGNMENT SITE is a literal `self.<prop> = ...` in the class or one of
    its ancestors, found with `ast`. Nothing else counts.

  * What this does NOT prove: that the assignment always runs. A site inside an
    `if` still counts as a site. Whether a subclass ever populates an inherited
    optional is dataflow — the hole this guard's JS twin documents around
    `details` — and reviewers still own that.

  * A property assigned any other way (setattr, a helper that writes onto the
    instance from outside, a base class outside the scanned files) is REPORTED,
    not guessed at. The JS side has a live example of exactly that shape:
    `AetherfyVectorsError.code` is stamped by `createErrorFromResponse`, not by
    a constructor, so documenting `err.code` there would report. If that ever
    becomes a false red here, the fix is to assign the property directly in the
    class, not to widen this check into a dataflow engine.

WIRING: this file is in tests/, so it runs in the same lane as every other
README guard — .github/workflows/test.yml, whose path filters cover README.md,
tests/** and both packages. test_readme_guard.py already asserts that README.md
is in both filter blocks; one rule, one gate, so that assertion is not repeated
here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

# The files that define this package's exception classes. Deliberately an
# explicit short list rather than a walk of the packages: aetherfy_vectors/
# schema.py defines an unrelated `ValidationError` dataclass, and a walk would
# collide it with the exception of the same name. test_every_error_class_the_
# readme_imports_is_in_the_scanned_sources below is what keeps this list honest
# if the exceptions ever move.
ERROR_SOURCES: Tuple[str, ...] = (
    "aetherfy_vectors/exceptions.py",
    "aetherfy_memory/exceptions.py",
)

# Where the ancestor walk stops. BaseException supplies `args` with no
# `self.args = ` anywhere in this repository, so it is named explicitly rather
# than left to read as MISSING. Everything else an exception "has" (`__cause__`,
# `__traceback__`) is dunder and unreachable through the claim form above.
NATIVE_ROOTS: Dict[str, Set[str]] = {
    "Exception": {"args"},
    "BaseException": {"args"},
    "object": set(),
}

# The README documents five error properties today. The floor is a tripwire on
# the EXTRACTOR, not a cap on the README: parsing fewer than this means the
# error-handling section moved, renamed its handler variable, or stopped being
# a ```python fence, and a silent zero would read as "all claims hold".
MIN_CLAIMS = 5

FENCE_RE = re.compile(r"^```(\w+)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def read_readme() -> str:
    return README.read_text(encoding="utf-8")


def live_sources() -> Dict[str, str]:
    """The error-class sources, keyed by repo-relative path.

    Returned as text rather than imported objects so the mutation proofs can
    hand this function's output back with one assignment removed.
    """
    return {rel: (REPO_ROOT / rel).read_text(encoding="utf-8") for rel in ERROR_SOURCES}


def python_fences(markdown: str) -> List[str]:
    """Every ```python block, dedented enough for `ast` to parse."""
    out = []
    for lang, body in FENCE_RE.findall(markdown):
        if lang != "python":
            continue
        lines = body.split("\n")
        indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
        strip = min(indents) if indents else 0
        out.append("\n".join(ln[strip:] if len(ln) >= strip else ln for ln in lines))
    return out


# ---------------------------------------------------------------------------
# The assignment sites
# ---------------------------------------------------------------------------


class ClassInfo(NamedTuple):
    # No `name` field: the table is keyed by class name, and a second copy of it
    # here was written by build_class_table and read by nothing.
    bases: Tuple[str, ...]
    props: Dict[str, str]  # property -> "path:line" of its assignment
    where: str


def _base_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):  # exceptions.AetherfyVectorsException
        return node.attr
    return None


def _self_assignments(node: ast.ClassDef, label: str) -> Dict[str, str]:
    """`self.<prop> = ` sites belonging to THIS class.

    The walk stops at a nested ClassDef: crediting an inner class's assignment
    to the outer one would be a fail-GREEN, the only direction that matters
    here. Nested classes are registered as their own entries by the caller.
    """
    lines: Dict[str, int] = {}
    stack: List[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        sub = stack.pop()
        if isinstance(sub, ast.ClassDef):
            continue
        targets: List[ast.expr] = []
        if isinstance(sub, ast.Assign):
            targets = list(sub.targets)
        elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
            targets = [sub.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                # First site by line number, so the report is stable no matter
                # what order the walk happened to reach them in.
                line = getattr(sub, "lineno", 0)
                if target.attr not in lines or line < lines[target.attr]:
                    lines[target.attr] = line
        stack.extend(ast.iter_child_nodes(sub))
    return {prop: f"{label}:{line}" for prop, line in lines.items()}


def build_class_table(sources: Dict[str, str]) -> Dict[str, ClassInfo]:
    """class name -> its bases and its `self.<prop> = ` sites.

    Reads assignments only. A property that is merely READ (``if
    self.retry_after:`` in ``__str__``) is not a site, or a class could
    document a property it only ever reads back as None.
    """
    table: Dict[str, ClassInfo] = {}
    for label, text in sources.items():
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            props = _self_assignments(node, label)
            bases = tuple(
                name for name in (_base_name(b) for b in node.bases) if name is not None
            )
            if node.name in table:
                raise AssertionError(
                    f"two classes named {node.name} in the scanned error sources "
                    f"({table[node.name].where} and {label}:{node.lineno}). A claim "
                    f"about {node.name} would be checked against whichever won, "
                    f"which is a coin toss, not a guard. Narrow ERROR_SOURCES or "
                    f"rename."
                )
            table[node.name] = ClassInfo(bases, props, f"{label}:{node.lineno}")
    return table


def inherited_props(
    table: Dict[str, ClassInfo], class_name: str
) -> Tuple[Dict[str, str], Optional[str]]:
    """Every property visible on `class_name`, walking its ancestors.

    The second return value is non-None when the chain could not be resolved —
    a base that is neither a scanned class nor a known native root. Saying
    MISSING off an incomplete chain would be a guess, so that is reported as
    its own problem instead.
    """
    props: Dict[str, str] = {}
    problem: Optional[str] = None
    seen: Set[str] = set()
    queue = [class_name]
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        if name in NATIVE_ROOTS:
            for prop in NATIVE_ROOTS[name]:
                props.setdefault(prop, f"native {name}")
            continue
        info = table.get(name)
        if info is None:
            problem = (
                f"{class_name} inherits from {name}, which is neither defined in "
                f"{' / '.join(ERROR_SOURCES)} nor a known native base — the "
                f"ancestor chain cannot be resolved, so this guard cannot say "
                f"whether a property is assigned. Add the file to ERROR_SOURCES."
            )
            continue
        for prop, site in info.props.items():
            props.setdefault(prop, site)
        queue.extend(info.bases)
    return props, problem


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------


class Claim(NamedTuple):
    cls: str
    prop: str
    where: str


def readme_claims(markdown: str, known_classes: Set[str]) -> List[Claim]:
    """Every `except <OurError> as e:` / `e.<prop>` pair, in document order.

    A tuple handler (`except (A, B) as e:`) claims the property against BOTH
    classes, because the body runs for either — that is the promise the reader
    is handed, not a choice between two.
    """
    claims: List[Claim] = []
    for index, fence in enumerate(python_fences(markdown), 1):
        try:
            tree = ast.parse(fence)
        except SyntaxError:
            # An unparseable fence is already red in test_readme_guard.py; this
            # guard has nothing to add and must not double-report it.
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not node.name:
                continue
            named: List[str] = []
            if isinstance(node.type, ast.Name):
                named = [node.type.id]
            elif isinstance(node.type, ast.Tuple):
                named = [e.id for e in node.type.elts if isinstance(e, ast.Name)]
            ours = [name for name in named if name in known_classes]
            if not ours:
                continue  # `except Exception as e:` is the reader's own business
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == node.name
                ):
                    for cls in ours:
                        claims.append(
                            Claim(cls, sub.attr, f"README.md python fence #{index}")
                        )
    # Deduplicated, because the same handler printing `e.limit` twice is one
    # promise; ordering is stabilised for readable failure output.
    return sorted(set(claims))


def audit(
    markdown: str, table: Dict[str, ClassInfo]
) -> Tuple[List[str], List[Claim]]:
    """Returns (problems, claims). Zero claims is itself a problem."""
    claims = readme_claims(markdown, set(table))
    if not claims:
        return (
            [
                "no error-property claims were parsed from the README at all. "
                "That is a failure, not a pass: an error-handling section that "
                "moved, renamed its handler variable, or stopped being a "
                "```python fence would otherwise read as 'every claim holds'."
            ],
            claims,
        )

    problems: List[str] = []
    for claim in claims:
        props, chain_problem = inherited_props(table, claim.cls)
        if chain_problem is not None:
            problems.append(f"{claim.where}: {chain_problem}")
            continue
        if claim.prop not in props:
            problems.append(
                f"{claim.where}: `{claim.cls}.{claim.prop}` is documented, but no "
                f"`self.{claim.prop} = ` exists in {claim.cls} or its ancestors "
                f"({' -> '.join(_chain(table, claim.cls))}). A reader's handler "
                f"would raise AttributeError. Assign it, or stop documenting it."
            )
    return problems, claims


def _chain(table: Dict[str, ClassInfo], class_name: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    queue = [class_name]
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        info = table.get(name)
        if info is not None:
            queue.extend(info.bases)
    return out


def claims_table(markdown: str, table: Dict[str, ClassInfo]) -> str:
    """The report artifact: property -> assignment site, or MISSING."""
    lines = []
    for claim in readme_claims(markdown, set(table)):
        props, _ = inherited_props(table, claim.cls)
        site = props.get(claim.prop, "MISSING")
        lines.append(f"{claim.cls}.{claim.prop} -> {site}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_every_documented_error_property_has_an_assignment_site():
    table = build_class_table(live_sources())

    # Refuse to trust an extraction that found nothing: a table with no classes
    # agrees with every claim by having nothing to disagree with.
    assert len(table) >= 15, (
        f"only {len(table)} classes parsed from {', '.join(ERROR_SOURCES)} — the "
        f"assignment-site extractor is blind, not satisfied"
    )
    assert "retry_after" in table["RateLimitExceededError"].props, (
        "the extractor did not find RateLimitExceededError.retry_after, which is "
        "assigned directly in exceptions.py — it is reading nothing useful"
    )

    problems, claims = audit(read_readme(), table)

    assert len(claims) >= MIN_CLAIMS, (
        f"only {len(claims)} error-property claims parsed from README.md, expected "
        f"at least {MIN_CLAIMS} — the claim parser has gone blind"
    )
    assert not problems, (
        "README documents error properties that nothing assigns:\n  "
        + "\n  ".join(problems)
        + "\n\nEvery claim, and where it resolves:\n"
        + claims_table(read_readme(), table)
    )


def test_a_planted_claim_is_caught():
    """Positive control. The zero above is worthless until the check fires.

    A README can document a property no code assigns; that is the whole reason
    this file exists. So plant exactly that and require a red naming it.
    """
    table = build_class_table(live_sources())
    original = read_readme()
    planted = original.replace(
        '    print(f"Rate limit exceeded. Retry after: {e.retry_after}s")',
        '    print(f"Rate limit exceeded. Retry after: {e.retry_after}s")\n'
        "    print(e.retry_after_seconds)",
        1,
    )
    assert planted != original, (
        "the planted claim did not apply — the anchor line moved, so this control "
        "proved nothing"
    )

    problems, _ = audit(planted, table)
    assert any(
        "RateLimitExceededError" in p and "retry_after_seconds" in p for p in problems
    ), (
        "a documented property with no assignment site did NOT red. The check is "
        f"decoration. Problems reported: {problems}"
    )

    # ...and the unplanted README stays green, or the control proves only noise.
    clean, _ = audit(original, table)
    assert clean == [], f"the check flags the real README: {clean}"


def test_zero_claims_is_a_failure_not_a_pass():
    """Vacuity guard. A section that moved must never read as 'all claims hold'."""
    table = build_class_table(live_sources())
    problems, claims = audit("# Aetherfy\n\nNo fenced samples here.\n", table)
    assert claims == []
    assert problems and "no error-property claims" in problems[0], (
        f"a README with zero claims passed the audit: {problems}"
    )

    # The realistic version of the same accident: the section is still there but
    # the handler variable is no longer bound, so nothing is claimed.
    unbound = (
        "```python\n"
        "try:\n"
        "    client.search('c', [0.1])\n"
        "except RateLimitExceededError:\n"
        "    print('rate limited')\n"
        "```\n"
    )
    problems, claims = audit(unbound, table)
    assert claims == []
    assert problems, "an error section that claims nothing must still be red"


def test_deleting_a_real_assignment_turns_the_check_red():
    """Mutation proof, asserted-applied before the verdict is trusted."""
    sources = live_sources()
    label = "aetherfy_vectors/exceptions.py"
    original = sources[label]

    needle = "        self.retry_after = retry_after\n"
    assert needle in original, (
        "the mutation target has moved; this proof would silently check nothing"
    )
    mutated = original.replace(needle, "", 1)
    assert mutated != original

    mutated_table = build_class_table({**sources, label: mutated})
    # The mutation LANDED — the site is gone from the table, not merely from the
    # text. `__str__` still reads self.retry_after; a read is not a site.
    assert "retry_after" not in mutated_table["RateLimitExceededError"].props, (
        "the mutation did not land: the extractor still reports an assignment "
        "site for retry_after, so the red below would not be caused by it"
    )

    problems, _ = audit(read_readme(), mutated_table)
    assert any(
        "RateLimitExceededError" in p and "retry_after" in p for p in problems
    ), f"deleting a real assignment did not red the guard: {problems}"

    # Restore -> green.
    restored, _ = audit(read_readme(), build_class_table(sources))
    assert restored == [], f"the unmutated tree is not green: {restored}"


def test_a_read_is_not_an_assignment_site():
    """The crux: reading a property back must not count as promising it."""
    table = build_class_table(
        {
            "synthetic.py": (
                "class Base(Exception):\n"
                "    def __init__(self, message):\n"
                "        super().__init__(message)\n"
                "        self.message = message\n"
                "\n"
                "class Reader(Base):\n"
                "    def __str__(self):\n"
                "        return f'{self.never_assigned}'\n"
            )
        }
    )
    props, chain_problem = inherited_props(table, "Reader")
    assert chain_problem is None
    assert "message" in props
    assert "never_assigned" not in props, (
        "a property that is only ever READ was counted as assigned — that is the "
        "exact bug shape this guard exists to catch"
    )


def test_a_nested_class_does_not_lend_its_assignments_to_the_outer_one():
    """Fail-green check on the extractor itself.

    A walk that descended into a nested class would credit the outer class with
    the inner one's `self.x = `, which reads as "assigned" for a property the
    outer class never sets — green, and wrong.
    """
    table = build_class_table(
        {
            "synthetic.py": (
                "class Outer(Exception):\n"
                "    class Inner:\n"
                "        def __init__(self):\n"
                "            self.inner_only = 1\n"
                "\n"
                "    def __init__(self):\n"
                "        self.outer_only = 2\n"
            )
        }
    )
    assert set(table["Outer"].props) == {"outer_only"}
    assert set(table["Inner"].props) == {"inner_only"}


def test_an_unresolvable_ancestor_is_reported_not_assumed():
    table = build_class_table(
        {"synthetic.py": ("class Orphan(SomethingElse):\n    pass\n")}
    )
    props, chain_problem = inherited_props(table, "Orphan")
    assert props == {}
    assert chain_problem is not None and "SomethingElse" in chain_problem


def test_duplicate_class_names_are_refused():
    with pytest.raises(AssertionError, match="two classes named ValidationError"):
        build_class_table(
            {
                "a.py": "class ValidationError(Exception):\n    pass\n",
                "b.py": "class ValidationError(Exception):\n    pass\n",
            }
        )


def test_every_error_class_the_readme_imports_is_in_the_scanned_sources():
    """Coverage pin for ERROR_SOURCES itself.

    A class the table does not know is not checked and is not reported either —
    `readme_claims` only claims against known class names, so moving an
    exception to a new file would drop its claims SILENTLY. That is the one
    fail-green this design has, and this is the assertion that closes it.
    """
    table = build_class_table(live_sources())
    imported: Set[str] = set()
    for fence in python_fences(read_readme()):
        try:
            tree = ast.parse(fence)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith(("aetherfy_vectors", "aetherfy_memory")):
                continue
            for alias in node.names:
                if alias.name.endswith(("Error", "Exception")):
                    imported.add(alias.name)

    assert len(imported) >= 10, (
        f"only {len(imported)} error classes found in the README's imports — the "
        f"import scan is dead and the assertion below is vacuous"
    )
    missing = sorted(name for name in imported if name not in table)
    assert not missing, (
        f"the README documents {', '.join(missing)}, which {', '.join(ERROR_SOURCES)} "
        f"do not define. Claims about those classes are being dropped silently. "
        f"Add the defining file to ERROR_SOURCES."
    )
