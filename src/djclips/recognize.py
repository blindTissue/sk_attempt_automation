"""shazam-api.com client wrapped in a persistent cache and a call budget.

Every paid call is cached on disk (keyed by the input file's fingerprint and
the snippet position), so re-running on the same mix never pays twice."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from urllib.parse import urljoin

from .audio import extract_mp3
from .segment import Probe

log = logging.getLogger(__name__)

DEFAULT_URL = "https://shazam-api.com/api/v2/recognize"


@dataclass
class Track:
    title: str
    artist: str
    isrc: str | None = None
    cover_url: str | None = None
    timecode: float | None = None
    links: dict = field(default_factory=dict)
    uid: str | None = None  # id in the editable tracks.json, when it came from there

    @property
    def key(self) -> str:
        if self.uid:
            return self.uid
        if self.isrc:
            return self.isrc.upper()
        return f"{_norm(self.artist)}|{_norm(self.title)}"

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.lower()).strip()


class BudgetExhausted(RuntimeError):
    pass


class ShazamClient:
    def __init__(self, api_key: str, url: str = DEFAULT_URL, field_name: str = "file", timeout: float = 90):
        self.url = url
        self.field_name = field_name
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def recognize_file(self, path: Path) -> dict:
        # Retry transport errors / 5xx and wait out 429 rate limits; none of these are billed.
        failures = 0
        while True:
            try:
                with open(path, "rb") as fh:
                    resp = self.session.post(
                        self.url, files={self.field_name: (path.name, fh, "audio/mpeg")}, timeout=self.timeout,
                    )
            except requests.ConnectionError:
                failures += 1
                if failures >= 3:
                    raise
                time.sleep(2 * failures)
                continue
            if resp.status_code == 429:
                delay = _retry_after(resp)
                log.info("rate limited; waiting %.0fs", delay)
                time.sleep(delay)
                continue
            if resp.status_code >= 500:
                failures += 1
                if failures >= 3:
                    resp.raise_for_status()
                time.sleep(2 * failures)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Shazam API error {resp.status_code}: {resp.text[:300]}")
            return resp.json()

    def wait(self, data: dict, poll: float = 2, timeout: float = 300) -> dict:
        """Recognition is asynchronous: poll `resultsUrl` until the job is done.
        Polling an existing job does not start (or pay for) a new recognition."""
        deadline = time.monotonic() + timeout
        while is_pending(data):
            if time.monotonic() > deadline:
                raise TimeoutError(f"recognition job still {data.get('status')!r} after {timeout:.0f}s")
            time.sleep(poll)
            resp = self.session.get(urljoin(self.url, data["resultsUrl"]), timeout=self.timeout)
            if resp.status_code >= 500:
                continue
            resp.raise_for_status()
            data = {**resp.json(), "resultsUrl": data["resultsUrl"]}
        return data


def _retry_after(resp: requests.Response) -> float:
    header = resp.headers.get("Retry-After", "")
    if header.isdigit():
        return int(header) + 1
    m = re.search(r"retry in (\d+) seconds", resp.text)
    return int(m.group(1)) + 1 if m else 30


PENDING = {"processing", "queued", "pending", "running"}


def is_pending(data: dict) -> bool:
    return data.get("status") in PENDING and bool(data.get("resultsUrl"))


def parse_response(data: dict) -> Track | None:
    """Pull the best match out of a response, tolerating schema variations."""
    results = data.get("results", data.get("result", data.get("track")))
    if isinstance(results, list):
        results = results[0] if results else None
    if not isinstance(results, dict):
        return None
    title = results.get("title")
    artist = results.get("artist") or results.get("subtitle") or results.get("artists")
    if isinstance(artist, list):
        artist = ", ".join(a.get("name", "") if isinstance(a, dict) else str(a) for a in artist)
    if not title:
        return None
    return Track(
        title=str(title),
        artist=str(artist or "Unknown artist"),
        isrc=results.get("isrc"),
        cover_url=_find_cover_url(results),
        timecode=results.get("timecode"),
        links=results.get("links") or {},
    )


def _find_cover_url(r: dict) -> str | None:
    for key in ("cover", "coverart", "cover_url", "artwork", "artwork_url", "image", "thumbnail"):
        v = r.get(key)
        if isinstance(v, str) and v.startswith("http"):
            return re.sub(r"/\d+x\d+(\w*)\.(jpg|png|webp)$", r"/1200x1200\1.\2", v)  # mzstatic: ask for hi-res
    images = r.get("images")
    if isinstance(images, dict):
        for key in ("coverarthq", "coverart", "background"):
            if isinstance(images.get(key), str):
                return images[key]
    return None


def file_fingerprint(path: Path) -> str:
    """Cheap identity: size + hash of the first and last 4 MB."""
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(4 << 20))
        if size > 8 << 20:
            fh.seek(-(4 << 20), 2)
            h.update(fh.read())
    return h.hexdigest()[:16]


class Recognizer:
    """Answers "what is playing in this probe?" from the cache when possible,
    otherwise (budget permitting) from the paid API."""

    def __init__(self, audio_path: Path, cache_path: Path, client: ShazamClient | None,
                 max_calls: int | None = None, dry_run: bool = False):
        self.audio_path = audio_path
        self.cache_path = cache_path
        self.client = client
        self.max_calls = max_calls
        self.dry_run = dry_run
        self.calls = 0
        self.fingerprint = file_fingerprint(audio_path)
        self.cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def _key(self, probe: Probe) -> str:
        return f"{self.fingerprint}@{probe.start:.1f}+{probe.duration:.1f}"

    def is_cached(self, probe: Probe) -> bool:
        return self._key(probe) in self.cache

    def recognize(self, probe: Probe) -> Track | None:
        key = self._key(probe)
        if key in self.cache and not is_pending(self.cache[key]):
            return parse_response(self.cache[key])
        if self.dry_run or self.client is None:
            return None
        if key in self.cache:  # submitted (and paid for) earlier but never collected
            return self._finish(key, self.cache[key])
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExhausted(f"call budget of {self.max_calls} reached")
        with tempfile.TemporaryDirectory() as tmp:
            snippet = extract_mp3(self.audio_path, probe.start, probe.duration, Path(tmp) / "snippet.mp3")
            self.calls += 1
            log.info("API call #%d: %.1fs-%.1fs", self.calls, probe.start, probe.end)
            data = self.client.recognize_file(snippet)
        self._store(key, data)  # persist the job id at once so a crash never means paying again
        return self._finish(key, data)

    def _finish(self, key: str, data: dict) -> Track | None:
        data = self.client.wait(data)
        self._store(key, data)
        return parse_response(data)

    def _store(self, key: str, data: dict) -> None:
        self.cache[key] = data
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False))
