from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .pipeline import Options, run
from .render import FORMATS


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    defaults = Options()
    p = argparse.ArgumentParser(prog="djclips", description="Make an Instagram-ready video of every transition in a DJ mix.")
    p.add_argument("input", type=Path, help="mix recording (.wav or .mp3)")
    p.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory (default: out)")
    p.add_argument("--api-key", default=os.environ.get("SHAZAM_API_KEY"), help="shazam-api.com key (or env SHAZAM_API_KEY)")
    p.add_argument("--api-url", default=os.environ.get("SHAZAM_API_URL", defaults.api_url))
    p.add_argument("--api-field", default=os.environ.get("SHAZAM_API_FIELD", defaults.api_field),
                   help="multipart field name for the audio upload (default: file)")
    p.add_argument("--dry-run", action="store_true", help="analyse and show planned API calls without making any")
    p.add_argument("--max-calls", type=int, default=None, help="hard cap on paid API calls this run")
    p.add_argument("--probe-seconds", type=float, default=defaults.probe_seconds, help="snippet length sent to the API")
    p.add_argument("--min-track", type=float, default=defaults.min_track, help="shortest plausible track in seconds")
    p.add_argument("--max-segment", type=float, default=defaults.max_segment,
                   help="add an extra probe for every this-many seconds of an unbroken segment")
    p.add_argument("--pad", type=float, default=defaults.pad, help="seconds kept before/after each transition")
    p.add_argument("--max-clip", type=float, default=defaults.max_clip, help="maximum video length in seconds")
    p.add_argument("--format", nargs="+", choices=list(FORMATS), default=list(defaults.formats),
                   help="video aspect ratio(s): 9x16 (Reels) and/or 4x5 (feed); default: both")
    p.add_argument("--tracks", type=Path, default=None,
                   help="editable track names/covers file (default: <out>/tracks.json)")
    p.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    if not a.input.exists():
        p.error(f"{a.input} not found")
    if not a.api_key and not a.dry_run:
        logging.warning("No API key: only cached recognitions will be used.")
    run(a.input, a.out, Options(
        api_key=a.api_key, api_url=a.api_url, api_field=a.api_field, cache_dir=a.cache_dir,
        dry_run=a.dry_run, max_calls=a.max_calls, probe_seconds=a.probe_seconds, min_track=a.min_track,
        max_segment=a.max_segment, pad=a.pad, max_clip=a.max_clip, formats=tuple(a.format), tracklist=a.tracks,
    ))


if __name__ == "__main__":
    main()
