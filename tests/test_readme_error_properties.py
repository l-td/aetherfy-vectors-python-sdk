"""Guards the README's error-object promises against the classes that raise.

WHY THIS EXISTS: the README tells integrators what a caught exception carries —
`e.retry_after`, `e.quota_type`, `e.errors`. Nothing tied those claims to the
classes. The README could document a property no constructor ever assigns and
every suite in this repository would stay green while the reader's handler
printed `None`. Publication makes the README the PyPI page, so those claims
become contractual; this is the check that says they are true.

It is the twin of tests/test_readme_guard.py, which checks the SAMPLES against
the API. That guard checks calls; this one checks the SHAPE OF WHAT IS CAUGHT.

THE CHECK has three rules, and a claim must pass all three:

  1. RESOLVES — the documented property has an assignment site (a literal
     `self.<prop> = `) somewhere in the class or its ancestors.
  2. RESOLVES CONCRETELY — that site is in the class the README NAMED, not
     merely inherited from a base. See below; this is the rule that catches the
     bug the whole README-guards arc exists because of.
  3. NAMES A CLASS WE CAN SEE — a claim about a class missing from the scanned
     sources REFUSES, red. It is never skipped, because a skipped claim is
     coverage lost in silence.

RULE 2, and why it is not paranoia. The defect that started this arc (recorded
in the JS twin's header) was a README logging `error.details` on a
SchemaValidationError. `details` IS declared on the base and IS assigned by the
base constructor — so rule 1 passes it — but SchemaValidationError never passes
`details` up, so it is always empty and the violations the reader wanted live in
`errors`. Requiring the site to be on the concrete class catches that
statically, with no dataflow: the base's `self.details = ` is not
SchemaValidationError promising anything.

  * BASE_PROPERTY_ALLOWLIST is the escape hatch for a base property that IS
    populated for every subclass. It is EMPTY today — every property the README
    documents resolves on the concrete class or to the native allowance below —
    and it is rot-checked, so an entry that stops being needed reds.

SCOPE, stated honestly:

  * A CLAIM is `except <OurError> as e:` inside a README ```python fence, plus
    an `e.<prop>` read anywhere in that handler's body. "Ours" means the README
    itself imported the name from one of OUR_PACKAGES in some fence — that is
    what separates our classes from `except Exception as e:`, without a naming
    convention that a future class could fail to follow.

  * Prose is deliberately not parsed: prose is ambiguous about whose property
    it names — the Limits section's "a structured `error.code`" is the SERVER
    envelope's key, not an attribute of any class in this package, and a parser
    that read it as one would red on a true sentence.

  * What this still does NOT prove: that the assignment always runs. A site
    inside an `if` counts as a site. Rule 2 removes the inherited-optional case,
    which is the one that has actually bitten; a concrete class that assigns a
    property only on some paths is still the reviewer's problem.

  * A property assigned any other way (setattr, a helper writing onto the
    instance from outside, a base outside the scanned files) is REPORTED, not
    guessed at. The JS side has a live example: `AetherfyVectorsError.code` is
    stamped by `createErrorFromResponse`, not by a constructor. If that ever
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
# collide it with the exception of the same name. Two things keep this list
# honest: a claim about a class missing from it REFUSES (rule 3), and
# test_every_error_class_the_readme_imports_is_in_the_scanned_sources reds even
# for classes no claim has been made about yet.
ERROR_SOURCES: Tuple[str, ...] = (
    "aetherfy_vectors/exceptions.py",
    "aetherfy_memory/exceptions.py",
)

# Packages whose imported names the README is understood to be documenting.
OUR_PACKAGES = ("aetherfy_vectors", "aetherfy_memory")

# Where the ancestor walk stops. BaseException supplies `args` with no
# `self.args = ` anywhere in this repository, so it is named explicitly rather
# than left to read as MISSING. Everything else an exception "has" (`__cause__`,
# `__traceback__`) is dunder and unreachable through the claim form above.
NATIVE_ROOTS: Dict[str, Set[str]] = {
    "Exception": {"args"},
    "BaseException": {"args"},
    "object": set(),
}

# Base-class properties the README may document on a SUBCLASS, because the base
# populates them for every subclass on every path that raises. Each entry needs
# a reason; all of them are rot-checked by
# test_the_base_property_allowlist_has_not_rotted, which reds on an entry that
# has stopped being needed.
#
# EMPTY, and that is the finding: every property the README documents today
# resolves on the concrete class it names. Adding an entry here is a deliberate
# act that says "this base property is genuinely populated for this subclass" —
# it is not the way to silence a red you do not understand.
BASE_PROPERTY_ALLOWLIST: Dict[str, str] = {}

# THE CENSUS. The exact set of error-object promises the README makes, as
# `Class.property`. This is a pinned expectation, not a floor: rewrite the error
# section into a different idiom — a table, `isinstance` inside a bare `except`,
# a walrus — and the parser stops seeing claims it used to see, which reds HERE
# as a census mismatch instead of quietly checking less than it did yesterday.
#
# To resolve a red: if the README genuinely documents a different set now,
# update this list in the same commit. That edit is the conscious act a count
# floor cannot force.
EXPECTED_CLAIMS: List[str] = [
    "QuotaExceededError.current",
    "QuotaExceededError.limit",
    "QuotaExceededError.quota_type",
    "RateLimitExceededError.retry_after",
    "SchemaValidationError.errors",
]

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


def _parsed_fences(markdown: str) -> List[Tuple[int, ast.Module]]:
    """(fence number, tree) for every python fence that parses.

    An unparseable fence is already red in test_readme_guard.py; this guard has
    nothing to add and must not double-report it.
    """
    out = []
    for index, fence in enumerate(python_fences(markdown), 1):
        try:
            out.append((index, ast.parse(fence)))
        except SyntaxError:
            continue
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


def readme_owned_names(markdown: str) -> Set[str]:
    """Every symbol the README imports from one of OUR_PACKAGES.

    This is how a claim about OUR class is told apart from `except Exception as
    e:` — by what the page itself imported, not by a naming convention a future
    class could fail to follow. It deliberately does not filter by an `Error`
    suffix: a handler is the only place these names are used as an except type,
    so nothing else can be mistaken for one.
    """
    owned: Set[str] = set()
    for _, tree in _parsed_fences(markdown):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in OUR_PACKAGES:
                    owned.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in OUR_PACKAGES:
                        owned.add((alias.asname or alias.name).split(".")[0])
    return owned


def readme_claims(markdown: str) -> List[Claim]:
    """Every `except <OurError> as e:` / `e.<prop>` pair, deduplicated.

    A tuple handler (`except (A, B) as e:`) claims the property against BOTH
    classes, because the body runs for either — that is the promise the reader
    is handed, not a choice between two.

    Note what is NOT filtered here: a handler naming one of our imported names
    produces a claim even when the class is absent from the scanned sources.
    `audit` refuses those. Dropping them here is exactly the silent coverage
    loss this guard is supposed to be immune to.
    """
    owned = readme_owned_names(markdown)
    claims: List[Claim] = []
    for index, tree in _parsed_fences(markdown):
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not node.name:
                continue
            named: List[str] = []
            if isinstance(node.type, ast.Name):
                named = [node.type.id]
            elif isinstance(node.type, ast.Tuple):
                named = [e.id for e in node.type.elts if isinstance(e, ast.Name)]
            ours = [name for name in named if name in owned]
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


def census(markdown: str) -> List[str]:
    """The claims as pinnable `Class.property` strings."""
    return sorted({f"{c.cls}.{c.prop}" for c in readme_claims(markdown)})


def audit(
    markdown: str, table: Dict[str, ClassInfo]
) -> Tuple[List[str], List[Claim]]:
    """Returns (problems, claims). Zero claims is itself a problem."""
    claims = readme_claims(markdown)
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
        info = table.get(claim.cls)

        # Rule 3 — a class we cannot see is refused, never skipped.
        if info is None:
            problems.append(
                f"{claim.where}: the README documents `{claim.cls}.{claim.prop}`, "
                f"but {claim.cls} is not defined in "
                f"{' / '.join(ERROR_SOURCES)}. This guard REFUSES rather than "
                f"skipping it: a skipped claim is coverage lost in silence. Add "
                f"the file that defines {claim.cls} to ERROR_SOURCES."
            )
            continue

        props, chain_problem = inherited_props(table, claim.cls)
        if chain_problem is not None:
            problems.append(f"{claim.where}: {chain_problem}")
            continue

        # Rule 1 — it resolves at all.
        site = props.get(claim.prop)
        if site is None:
            problems.append(
                f"{claim.where}: `{claim.cls}.{claim.prop}` is documented, but no "
                f"`self.{claim.prop} = ` exists in {claim.cls} or its ancestors "
                f"({' -> '.join(_chain(table, claim.cls))}). A reader's handler "
                f"would raise AttributeError. Assign it, or stop documenting it."
            )
            continue

        # Rule 2 — it resolves on the class the README named.
        if claim.prop in info.props:
            continue
        if site.startswith("native "):
            continue
        if claim.prop in BASE_PROPERTY_ALLOWLIST:
            continue
        problems.append(
            f"{claim.where}: `{claim.cls}.{claim.prop}` is documented, but "
            f"{claim.cls} never assigns it — the only site is inherited, at "
            f"{site}. That is the shape of the defect this guard exists for: "
            f"`details` was declared and assigned on the base, so it type-checked "
            f"and existed, but the subclass the README named never populated it "
            f"and readers got an empty value. Assign it on {claim.cls}, document "
            f"the property the class actually sets, or — only if the base truly "
            f"populates it for every subclass — add it to BASE_PROPERTY_ALLOWLIST "
            f"with a reason."
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
    for claim in readme_claims(markdown):
        props, _ = inherited_props(table, claim.cls)
        site = props.get(claim.prop, "MISSING")
        info = table.get(claim.cls)
        scope = "concrete" if info and claim.prop in info.props else "inherited"
        if site.startswith("native "):
            scope = "native"
        lines.append(f"{claim.cls}.{claim.prop} -> {site} ({scope})")
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

    problems, _ = audit(read_readme(), table)
    assert not problems, (
        "README documents error properties that nothing assigns:\n  "
        + "\n  ".join(problems)
        + "\n\nEvery claim, and where it resolves:\n"
        + claims_table(read_readme(), table)
    )


def test_the_claim_census_is_exactly_what_is_pinned():
    """The set, not a floor. See EXPECTED_CLAIMS for how to resolve a red."""
    assert census(read_readme()) == EXPECTED_CLAIMS


def test_the_census_reds_when_the_readme_documents_less():
    """Mutation proof for the census, asserted-applied first.

    The failure this defends against is a README rewrite into an idiom the
    parser does not recognise: the claims silently disappear and everything
    still passes. Deleting one handler is that failure in miniature.
    """
    original = read_readme()
    handler = (
        "except QuotaExceededError as e:\n"
        "    print(f\"Quota '{e.quota_type}' exceeded: {e.current}/{e.limit}\")\n"
    )
    assert handler in original, (
        "the mutation target has moved; this proof would silently check nothing"
    )
    shrunk = original.replace(handler, "", 1)
    assert shrunk != original

    parsed = census(shrunk)
    # The mutation LANDED — the three QuotaExceededError promises are gone.
    assert not any(c.startswith("QuotaExceededError.") for c in parsed), (
        f"the mutation did not land: {parsed}"
    )
    assert parsed != EXPECTED_CLAIMS, (
        "the census did not notice a README that documents three fewer "
        "properties — it cannot force the conscious update it exists for"
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


def test_the_founding_bug_replanted_is_caught():
    """Rule 2, proved against the defect the arc exists because of.

    `details` is declared on AetherfyVectorsException and assigned by its
    constructor, so it EXISTS on every subclass and rule 1 passes it. But
    SchemaValidationError never passes `details` up, so a reader who followed a
    README documenting `e.details` would get `{}` where the violations should
    be. That is the JS twin's recorded historical defect, in this package's
    spelling. It must red.
    """
    table = build_class_table(live_sources())
    original = read_readme()
    anchor = "except SchemaValidationError as e:\n    for violation in e.errors:"
    assert anchor in original, "the mutation target has moved"
    planted = original.replace(anchor, f"{anchor}\n        print(e.details)", 1)
    assert planted != original

    # The mutation LANDED as a real claim, not a typo the parser dropped.
    assert "SchemaValidationError.details" in census(planted), (
        "the planted read was not parsed as a claim, so the red below — if any — "
        "would not be the founding bug"
    )
    # And rule 1 alone would have PASSED it: the site genuinely exists.
    props, _ = inherited_props(table, "SchemaValidationError")
    assert "details" in props, (
        "premise broken: `details` no longer resolves at all, so this no longer "
        "reproduces a bug that rule 1 misses"
    )

    problems, _ = audit(planted, table)
    assert any(
        "SchemaValidationError" in p and "details" in p and "inherited" in p
        for p in problems
    ), f"the founding bug did not red: {problems}"


def test_the_base_property_allowlist_has_not_rotted():
    """An exemption that stopped being needed is an exemption that must go."""
    table = build_class_table(live_sources())
    needed: Set[str] = set()
    for claim in readme_claims(read_readme()):
        info = table.get(claim.cls)
        if info is None or claim.prop in info.props:
            continue
        props, _ = inherited_props(table, claim.cls)
        site = props.get(claim.prop)
        if site is None or site.startswith("native "):
            continue
        needed.add(claim.prop)

    stale = sorted(set(BASE_PROPERTY_ALLOWLIST) - needed)
    assert not stale, (
        f"BASE_PROPERTY_ALLOWLIST exempts {', '.join(stale)}, which no README "
        f"claim needs any more. Delete the entry — a stale exemption is a hole "
        f"held open for a reason that has expired."
    )
    assert all(reason.strip() for reason in BASE_PROPERTY_ALLOWLIST.values()), (
        "every allowlist entry needs a reason; an unexplained exemption is "
        "indistinguishable from a silenced red"
    )


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
        "from aetherfy_vectors.exceptions import RateLimitExceededError\n"
        "try:\n"
        "    client.search('c', [0.1])\n"
        "except RateLimitExceededError:\n"
        "    print('rate limited')\n"
        "```\n"
    )
    problems, claims = audit(unbound, table)
    assert claims == []
    assert problems, "an error section that claims nothing must still be red"


def test_a_claim_about_an_unseen_class_refuses():
    """Rule 3. The residual hole, closed: skipping is never the answer."""
    table = build_class_table(live_sources())
    markdown = (
        "```python\n"
        "from aetherfy_vectors.exceptions import BrandNewError\n"
        "try:\n"
        "    client.search('c', [0.1])\n"
        "except BrandNewError as e:\n"
        "    print(e.whatever)\n"
        "```\n"
    )
    problems, claims = audit(markdown, table)
    assert [f"{c.cls}.{c.prop}" for c in claims] == ["BrandNewError.whatever"], (
        "the claim was dropped instead of carried through to a refusal — that is "
        "the silent coverage loss this rule exists to prevent"
    )
    assert any("BrandNewError" in p and "REFUSES" in p for p in problems), (
        f"a claim about an unscanned class did not refuse: {problems}"
    )


def test_a_foreign_exception_is_not_claimed_against_us():
    """The other side of rule 3: `except Exception as e:` is not our business."""
    table = build_class_table(live_sources())
    markdown = (
        "```python\n"
        "try:\n"
        "    collections = client.get_collections()\n"
        "except Exception as e:\n"
        "    return {'status': 'unhealthy', 'error': str(e)}\n"
        "```\n"
    )
    _, claims = audit(markdown, table)
    assert claims == [], f"a builtin exception was claimed against our classes: {claims}"

    # The README's own health-check sample is exactly this shape, so the live
    # census is the standing proof that it stays out.
    assert not any(c.startswith("Exception.") for c in census(read_readme()))


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

    Distinct from rule 3, which refuses a CLAIM about an unseen class: this reds
    for a class the README imports but has not documented a property on yet, so
    a move is caught before the first claim is even made.
    """
    table = build_class_table(live_sources())
    imported = {
        name
        for name in readme_owned_names(read_readme())
        if name.endswith(("Error", "Exception"))
    }

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
