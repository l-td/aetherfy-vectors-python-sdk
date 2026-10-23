"""Regenerate tests/fixtures/request-bodies.json from the client's HTTP layer.

    python scripts/record_request_bodies.py

Run it after a change to a request the client builds; commit the file with the
change. tests/test_request_bodies_fixture.py fails while the file is stale.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import request_bodies  # noqa: E402


def main() -> int:
    rows = request_bodies.record()
    request_bodies.FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    request_bodies.FIXTURE.write_text(request_bodies.dump(rows), encoding="utf-8")
    print(f"wrote {len(rows)} request bodies to {request_bodies.FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
