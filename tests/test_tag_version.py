"""The release refuses a tag that is not the package version.

Runs scripts/check_tag_version.py the way .github/workflows/publish.yml does,
and holds the workflow to running it before anything is built.
"""

import subprocess
import sys
from pathlib import Path

import aetherfy_vectors

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_tag_version.py"
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"


def run(tag):
    return subprocess.run(
        [sys.executable, str(SCRIPT), tag], capture_output=True, text=True
    )


def test_the_matching_tag_passes():
    r = run(f"v{aetherfy_vectors.__version__}")
    assert r.returncode == 0, r.stderr


def test_control_a_mismatched_tag_fails():
    for tag in ["v9.9.9", aetherfy_vectors.__version__, "", "v1.2"]:
        r = run(tag)
        assert r.returncode == 1, tag


def test_the_workflow_checks_the_tag_before_building():
    text = WORKFLOW.read_text(encoding="utf-8")
    check = text.index('python scripts/check_tag_version.py "$GITHUB_REF_NAME"')
    assert check < text.index("python -m build")
    assert check < text.index("twine upload")
