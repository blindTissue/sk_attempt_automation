import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from djclips.features import ANALYSIS_SR, compute_features
from djclips.pipeline import Options, run
from djclips.recognize import Recognizer, ShazamClient, Track, parse_response
from djclips.segment import Probe, find_boundaries, plan_probes
from djclips.transitions import Occurrence, assign_unknowns, clip_window, group_occurrences, locate

SR = ANALYSIS_SR


def synth_track(notes, wave_shape, seconds, bpm, seed):
    """A looping melody with a distinct timbre plus a drum pattern."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    step = 60 / bpm / 2
    idx = (t // step).astype(int) % len(notes)
    freq = 440 * 2 ** ((np.array(notes)[idx] - 69) / 12)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    if wave_shape == "square":
        tone = np.sign(np.sin(phase)) * 0.3
    else:
        tone = sum(np.sin(k * phase) / k for k in range(1, 8)) * 0.3
    beat = (t % (60 / bpm)) < 0.05
    drums = beat * rng.normal(0, 0.4, len(t)) if wave_shape == "square" else beat * np.sin(2 * np.pi * 60 * t)
    return (tone + drums).astype(np.float32)


TRACK_NOTES = [
    ([60, 64, 67, 72, 67, 64, 65, 69], "square", 124),
    ([66, 69, 73, 78, 76, 73, 71, 69], "saw", 128),
    ([55, 58, 62, 63, 62, 58, 60, 51], "square", 126),
]


def synth_mix(lengths=(150, 160, 150), fade=24):
    """Tracks crossfaded over `fade` seconds; returns the mix and ground-truth transitions."""
    out = np.zeros(int((sum(lengths) - fade * (len(lengths) - 1)) * SR), dtype=np.float32)
    pos, truth = 0, []
    for i, (n, (notes, shape, bpm)) in enumerate(zip(lengths, TRACK_NOTES)):
        x = synth_track(notes, shape, n, bpm, i)
        env = np.ones_like(x)
        f = fade * SR
        if i > 0:
            env[:f] = np.linspace(0, 1, f)
            truth.append((pos / SR, pos / SR + fade))
        if i < len(lengths) - 1:
            env[-f:] = np.linspace(1, 0, f)
        out[pos:pos + len(x)] += x * env
        pos += len(x) - f
    return out, truth


def write_wav(path: Path, y: np.ndarray, sr=SR):
    pcm = (np.clip(y / (np.abs(y).max() + 1e-9), -1, 1) * 32000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


@pytest.fixture(scope="module")
def mix():
    y, truth = synth_mix()
    return compute_features(y), truth, len(y) / SR


def test_boundaries_found_near_each_transition(mix):
    F, truth, _ = mix
    b = find_boundaries(F)
    assert len(b) == len(truth)
    for bound, (s, e) in zip(b, truth):
        assert s - 10 <= bound <= e + 10


def test_one_probe_per_segment(mix):
    F, truth, total = mix
    probes = plan_probes(len(F), find_boundaries(F), clip=10)
    assert len(probes) == len(truth) + 1


def test_locate_transition(mix):
    F, truth, _ = mix
    s, e = truth[0]
    boundary = (s + e) / 2  # where segmentation puts it
    a = Occurrence(Track("A", "x"), [Probe(60, 10, 0, boundary)])
    b = Occurrence(Track("B", "y"), [Probe(e + 50, 10, boundary, e + 110)])
    start, cross, end = locate(F, a, b)
    assert abs(start - s) <= 6, (start, s)
    assert abs(end - e) <= 6, (end, e)
    assert start < cross < end


def test_unmatched_segment_is_resolved_by_sound(mix):
    F, truth, total = mix
    probes = plan_probes(len(F), find_boundaries(F), clip=10)
    a, b = Track("A", "x"), Track("B", "y")
    # The middle segment (track B) went unmatched: it doesn't sound like A or C → stays unknown.
    resolved = assign_unknowns(F, probes, [a, None, Track("C", "z")])
    assert resolved[1] is None
    # A second probe inside B's segment that went unmatched is resolved to B.
    s0, e0 = probes[1].segment
    extra = Probe(round(s0 + 60, 1), 10, s0, e0)
    resolved = assign_unknowns(F, [probes[0], probes[1], extra, probes[2]], [a, b, None, None])
    assert resolved[2] is b


def test_no_transition_across_unknown_track():
    p = [Probe(i * 100, 10, i * 100, i * 100 + 100) for i in range(3)]
    occ = group_occurrences(p, [Track("A", "x"), None, Track("C", "z")])
    assert [o.after_gap for o in occ] == [False, True]


def test_clip_window_caps_length():
    start, end, cs, ce = clip_window(100, 140, 200, 1000, pad=10, max_clip=60)
    assert ce - cs == pytest.approx(60)
    assert start <= 140 <= end
    _, _, cs, ce = clip_window(5, 8, 12, 30)
    assert (cs, ce) == (0, 22)


def test_parse_response_variants():
    t = parse_response({"status": "success", "results": [
        {"title": "CHANEL", "artist": "Tyla", "isrc": "USSM12503648", "timecode": 48.85}]})
    assert (t.title, t.artist, t.key) == ("CHANEL", "Tyla", "USSM12503648")
    assert parse_response({"status": "success", "results": []}) is None
    t = parse_response({"track": {"title": "X", "subtitle": "Y", "images": {"coverart": "https://c/1.jpg"}}})
    assert (t.artist, t.cover_url) == ("Y", "https://c/1.jpg")


class FakeClient:
    """Answers by looking up which synthetic track covers the snippet's time."""

    def __init__(self, spans):
        self.spans, self.calls = spans, 0

    def answer(self, start):
        self.calls += 1
        for (s, e), name in self.spans:
            if s <= start < e:
                return {"status": "success", "results": [{"title": f"Song {name}", "artist": f"Artist {name}"}]}
        return {"status": "success", "results": []}


@pytest.fixture
def fake_recognizer(monkeypatch):
    """Exercise the real Recognizer/cache code; only the HTTP call is faked.
    The fake 'snippet' just records its start time."""
    holder = {}

    def fake_extract(path, start, duration, out_path):
        out_path.write_text(str(start))
        return out_path

    monkeypatch.setattr("djclips.recognize.extract_mp3", fake_extract)
    monkeypatch.setattr(ShazamClient, "recognize_file",
                        lambda self, path: holder["client"].answer(float(path.read_text())))
    monkeypatch.setattr("djclips.pipeline.fetch_cover", lambda track, d: None)
    return holder


def test_end_to_end_makes_one_video_per_transition(tmp_path, fake_recognizer):
    y, truth = synth_mix()
    wav = tmp_path / "mix.wav"
    write_wav(wav, y)
    spans = [((0, truth[0][0] + 12), "A"), ((truth[0][0] + 12, truth[1][0] + 12), "B"), ((truth[1][0] + 12, 1e9), "C")]
    fake_recognizer["client"] = FakeClient(spans)
    opts = Options(api_key="k", cache_dir=tmp_path / "cache")

    report = run(wav, tmp_path / "out", opts)
    assert [t["title"] for t in report["tracks"]] == ["Song A", "Song B", "Song C"]
    assert len(report["transitions"]) == 2
    assert report["api_calls"] == 3
    for tr, (s, e) in zip(report["transitions"], truth):
        cs, ce = tr["clip"]
        assert ce - cs <= 60
        assert cs <= s - 4 and ce >= e + 4
        for fmt, (w, h) in (("9x16", (1080, 1920)), ("4x5", (1080, 1350))):
            mp4 = tmp_path / "out" / tr["videos"][fmt]
            info = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(mp4)],
                capture_output=True, text=True, check=True).stdout)
            v = next(s for s in info["streams"] if s["codec_type"] == "video")
            assert (v["width"], v["height"], v["codec_name"], v["pix_fmt"]) == (w, h, "h264", "yuv420p")
            assert abs(float(info["format"]["duration"]) - (ce - cs)) < 0.5

    # Second run is served entirely from cache: zero new API calls.
    assert run(wav, tmp_path / "out2", opts)["api_calls"] == 0

    # Manual edits in tracks.json win over Shazam and cost nothing.
    tl_path = tmp_path / "out" / "tracks.json"
    tl = json.loads(tl_path.read_text())
    a_id = tl["segments"][0]["track"]
    tl["tracks"][a_id].update(title="Fixed Title", artist="Fixed Artist", cover="mycover.png")
    from PIL import Image
    Image.new("RGB", (50, 50), "red").save(tmp_path / "out" / "mycover.png")
    tl_path.write_text(json.dumps(tl))
    report = run(wav, tmp_path / "out", opts)
    assert report["api_calls"] == 0
    assert report["tracks"][0]["title"] == "Fixed Title"
    assert report["transitions"][0]["from"]["cover_url"].endswith("mycover.png")
    assert sorted(p.name for p in (tmp_path / "out" / "4x5").glob("*.mp4"))[0].startswith("01_Fixed-Title_x_")
    assert len(list((tmp_path / "out" / "9x16").glob("*.mp4"))) == 2  # stale names removed

    # Reassigning a segment to the next track removes that transition.
    tl = json.loads(tl_path.read_text())
    tl["segments"][1]["track"] = a_id
    tl_path.write_text(json.dumps(tl))
    assert len(run(wav, tmp_path / "out", opts)["transitions"]) == 1


def test_async_job_is_polled_and_resumed_without_paying_twice(tmp_path, monkeypatch):
    monkeypatch.setattr("djclips.recognize.extract_mp3", lambda path, s, d, out: out)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 100)
    pending = {"apiVersion": 2, "status": "processing", "uuid": "u", "resultsUrl": "/api/v2/results/u"}
    done = {"apiVersion": 2, "status": "success", "results": [{"title": "T", "artist": "A"}]}

    class Client:
        submits = 0

        def recognize_file(self, path):
            Client.submits += 1
            return pending

        def wait(self, data):
            raise TimeoutError("still processing")

    rec = Recognizer(audio, tmp_path / "c.json", Client())
    with pytest.raises(TimeoutError):
        rec.recognize(Probe(0, 10))
    assert Client.submits == 1 and rec.is_cached(Probe(0, 10))  # job id persisted

    Client.wait = lambda self, data: done
    rec = Recognizer(audio, tmp_path / "c.json", Client())
    assert rec.recognize(Probe(0, 10)).title == "T"
    assert Client.submits == 1 and rec.calls == 0
