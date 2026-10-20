"""Refuse a release whose git tag is not the package's version.

Run by .github/workflows/publish.yml before anything is built:

    python scripts/check_tag_version.py "$GITHUB_REF_NAME"

The tag must be ``v`` + ``aetherfy_vectors.__version__`` (the one version of
all three packages; setup.py reads the same line). Exit 0 when they match,
1 when they do not, with both values printed.
"""

import ast
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parent.parent / "aetherfy_vectors" / "__init__.py"


def package_version() -> str:
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
    raise SystemExit(f"no __version__ string literal in {INIT}")


def main(argv) -> int:
    if len(argv) != 2 or not argv[1]:
        print("usage: check_tag_version.py <tag>", file=sys.stderr)
        return 1
    tag, version = argv[1], package_version()
    if tag != f"v{version}":
        print(
            f"The tag {tag!r} is not this package's version {version!r} "
            f"(expected 'v{version}'). Nothing is published.",
            file=sys.stderr,
        )
        return 1
    print(f"tag {tag} matches version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
