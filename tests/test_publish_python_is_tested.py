"""The release builds on a Python the test matrix runs.

.github/workflows/publish.yml builds and checks the package on one Python. If
that Python is not in .github/workflows/test.yml's matrix, the release runs on
a version nothing tests, or one past its end of life once the matrix moves on.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"


def publish_python(text):
    found = re.findall(r"^\s*python-version:\s*['\"]?([0-9.]+)['\"]?\s*$", text, re.M)
    assert len(found) == 1, f"publish.yml sets python-version {len(found)} times: {found}"
    return found[0]


def matrix_pythons(text):
    found = re.findall(r"^\s*python-version:\s*\[([^\]]*)\]", text, re.M)
    assert len(found) == 1, f"test.yml has {len(found)} python-version matrices"
    return [v.strip().strip("'\"") for v in found[0].split(",") if v.strip()]


def test_the_release_python_is_in_the_test_matrix():
    release = publish_python((WORKFLOWS / "publish.yml").read_text(encoding="utf-8"))
    tested = matrix_pythons((WORKFLOWS / "test.yml").read_text(encoding="utf-8"))
    assert release in tested, f"publish.yml builds on Python {release}; test.yml tests {tested}"


def test_a_release_python_outside_the_matrix_is_caught():
    tested = matrix_pythons('python-version: ["3.11", "3.12"]\n')
    assert publish_python("python-version: '3.10'\n") not in tested
