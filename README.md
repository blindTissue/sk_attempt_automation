# djclips

Takes a DJ mix (`.wav`/`.mp3`), finds every transition between tracks, and writes one
Instagram Reels video per transition. Each video covers 10 s before the transition through
10 s after it, at most 60 s in total. It shows a still of cover A × cover B with each title and artist.

## Setup

```sh
uv sync
cp .env.example .env   # then put your shazam-api.com key in it
```

Requires `ffmpeg`/`ffprobe` on PATH.

## Usage

```sh
uv run djclips mix.mp3 --dry-run        # analyse locally; prints the API calls it would make
uv run djclips mix.mp3 -o out           # make the videos
uv run djclips mix.mp3 --max-calls 15   # hard cap on paid calls
```

Output (H.264/AAC, 30 fps, faststart):
- `out/9x16/NN_<titleA>_x_<titleB>.mp4`: Reels, 1080x1920
- `out/4x5/NN_<titleA>_x_<titleB>.mp4`: feed, 1080x1350
- `out/tracks.json`: editable track names and covers (see below)
- `out/report.json`: tracklist, transition times, clip ranges, API calls used

Use `--format 9x16` or `--format 4x5` to render only one aspect ratio.

## Fixing names, covers or wrong matches

Edit `out/tracks.json` and re-run the same command. Recognitions are cached, so this costs
no API calls and only re-renders.

- `tracks.<id>.title` / `artist`: the text shown under the cover.
- `tracks.<id>.cover`: an image URL, or a local file path (relative to `tracks.json`).
- `segments[].track`: which track a part of the mix is. Set it to an id to fix a wrong or
  missing Shazam match, or to `null` for unknown. You can add new ids under `tracks`.
  This is how you recover a transition that was skipped because a segment went unmatched.

## How it saves API calls

1. **Local segmentation (free):** per-second chroma (melody/harmony) and MFCC (timbre)
   features → self-similarity novelty curve → candidate track boundaries.
2. **One probe per segment:** a 10 s snippet from the middle of each segment goes to the API.
   Segments longer than `--max-segment` (6 min) get one extra probe per 6 min. Consecutive
   probes that return the same track are merged.
3. **Persistent cache** (`~/.cache/djclips/shazam.json`): re-runs with different settings
   or output folders never pay twice for the same snippet.
4. **Transition placement is local:** each second between the two recognised snippets is
   compared to the melody/timbre of track A and track B, giving the transition's start,
   crossover and end.
5. Album covers come from the iTunes / Deezer public search APIs (free).

For a typical mix, expect roughly one call per track.

## Tuning

| flag | default | meaning |
|---|---|---|
| `--min-track` | 60 | shortest plausible track (s); raise it if the dry run shows too many probes |
| `--max-segment` | 360 | extra probe per this many seconds of unbroken segment |
| `--probe-seconds` | 10 | snippet length sent to the API |
| `--pad` / `--max-clip` | 10 / 60 | context around the transition / video length cap |
| `--api-field` | `file` | multipart field name for the upload, if your API plan differs |
