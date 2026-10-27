"""Record real Wikipedia REST summaries into wikipedia/<encoded tag>.json.

Usage: python record_wikipedia.py   (needs network access to *.wikipedia.org)
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import httpx

TAGS = [
    "en:Golden Gate Bridge",
    "en:Fort Point, San Francisco",
    "en:Palace of Fine Arts",
    "en:Crissy Field",
]
UA = "PointView/0.1 (https://github.com/kylephernandez/pointview/pointview; fixture recorder) httpx"


def fixture_name(tag: str) -> str:
    return urllib.parse.quote(tag, safe="").replace("%", "_") + ".json"


def main() -> None:
    out = Path(__file__).parent / "wikipedia"
    out.mkdir(exist_ok=True)
    with httpx.Client(headers={"User-Agent": UA}, timeout=15, follow_redirects=True) as c:
        for tag in TAGS:
            if (out / fixture_name(tag)).exists():
                continue
            lang, title = tag.split(":", 1)
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(
                title.replace(" ", "_"), safe=",()"
            )
            for attempt in range(20):  # shared egress IPs get rate limited (429/403): back off
                r = c.get(url)
                if r.status_code not in (403, 429):
                    break
                time.sleep(min(60, 10 * (attempt + 1)))
            r.raise_for_status()
            (out / fixture_name(tag)).write_text(
                json.dumps(r.json(), indent=1, ensure_ascii=False) + "\n"
            )
            print(tag, "->", fixture_name(tag))


if __name__ == "__main__":
    main()
