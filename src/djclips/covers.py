"""Album-art lookup using free catalogues (no paid calls)."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

import requests
from PIL import Image

from .recognize import Track, _norm

log = logging.getLogger(__name__)
TIMEOUT = 20


def fetch_cover(track: Track, cache_dir: Path) -> Image.Image | None:
    if track.cover_url and not track.cover_url.startswith(("http://", "https://")):
        local = Path(track.cover_url).expanduser()
        if local.exists():
            return Image.open(local).convert("RGB")
        log.warning("cover file not found: %s", local)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ident = f"{track.key}|{track.cover_url or ''}"  # a changed cover URL must not hit the old cache entry
    cached = cache_dir / (hashlib.sha1(ident.encode()).hexdigest()[:16] + ".jpg")
    if cached.exists():
        return Image.open(cached).convert("RGB")
    remote = track.cover_url if track.cover_url and track.cover_url.startswith("http") else None
    for source in (lambda: remote, lambda: _itunes(track), lambda: _deezer(track)):
        try:
            url = source()
            if not url:
                continue
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:  # any lookup failure just falls through to the next source
            log.debug("cover source failed for %s: %s", track.title, exc)
            continue
        img.save(cached, quality=95)
        return img
    log.warning("no cover found for %s - %s", track.artist, track.title)
    return None


def _artist_matches(want: str, got: str) -> bool:
    w, g = _norm(want), _norm(got)
    return bool(w) and (w in g or g in w or w.split()[0] in g)


def _itunes(track: Track) -> str | None:
    resp = requests.get("https://itunes.apple.com/search", timeout=TIMEOUT, params={
        "term": f"{track.artist} {track.title}", "entity": "song", "limit": 10})
    results = resp.json().get("results", [])
    best = next((r for r in results if _artist_matches(track.artist, r.get("artistName", ""))), None)
    best = best or next((r for r in results if _norm(r.get("trackName", "")) == _norm(track.title)), None)
    url = best and best.get("artworkUrl100")
    return url.replace("100x100bb", "1200x1200bb") if url else None


def _deezer(track: Track) -> str | None:
    resp = requests.get("https://api.deezer.com/search", timeout=TIMEOUT,
                        params={"q": f'artist:"{track.artist}" track:"{track.title}"'})
    data = resp.json().get("data", [])
    if not data:
        resp = requests.get("https://api.deezer.com/search", timeout=TIMEOUT,
                            params={"q": f"{track.artist} {track.title}"})
        data = resp.json().get("data", [])
    return data[0]["album"].get("cover_xl") if data else None
