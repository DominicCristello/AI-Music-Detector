"""
build_dataset.py
Builds EITHER half of the AI-vs-human dataset with identical codec augmentation:
  * main()               -> HUMAN half: stream lossless FLAC masters from Jamendo
  * augment_local_folder -> AI half (or any local folder): same variants, label=1
Keeping both in one file guarantees the two halves are processed identically --
same codecs, same feature extractor, same columns.

STREAMING pipeline: download a lossless FLAC master from Jamendo, immediately
expand it into 5 codec variants, extract features from each, append the rows to
the feature CSV, then delete the (large) variant files -- keeping only the FLAC
master. One command instead of "download everything, then process everything".

WHY variants (codec augmentation):
  Your features (freqExceeds10kHz, spectralRolloff, spectralFlatness, ...) are
  frequency-based and therefore VERY sensitive to lossy compression. If the human
  half is all lossless and a user later submits a 128kbps MP3, the model breaks.
  By training on the SAME song across many codecs (in BOTH the human and AI
  halves), codec stops being a shortcut for the label, and the model generalises
  to whatever format a real user throws at it.

HOW IT REUSES YOUR CODE:
  * fetch_human_tracks.py  -> Jamendo config, license filter, provenance writer
  * process_folder.py      -> _process_single_file (the real feature extractor),
                              FIELD_MAP, write_batch, find_audio_files
  Nothing about feature computation is re-implemented here.

DISK MODEL:
  masters kept forever in MASTER_DIR (so you can regenerate variants later without
  re-downloading). Variants are written to SCRATCH_DIR and deleted immediately
  after their features are extracted, so peak variant disk use is tiny.

REQUIRES: ffmpeg on PATH (you have it). The script calls the ffmpeg executable via
  subprocess -- no path configuration needed.

RESUME: re-run anytime. A (baseTrackId, codec) pair already in the output CSV is
  skipped; a fully-finished song is not re-downloaded.

TRAIN/TEST SPLIT WARNING: every row carries baseTrackId + codec columns. When you
  split, GROUP BY baseTrackId so all 5 codec variants of a song stay on the SAME
  side. Otherwise near-duplicate leakage inflates your accuracy.
"""

import os
import csv
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import requests

# reuse the downloader's config + helpers (importing does NOT run its main())
import fetch_human_tracks as fetch
# reuse the REAL feature extractor + CSV plumbing
from process_folder import _process_single_file, FIELD_MAP, write_batch

# ── config ────────────────────────────────────────────────────────────────────
MASTER_DIR  = r"G:\AI_Music_Detection_Library\Human_FLAC_Masters"   # kept forever
SCRATCH_DIR = r"G:\AI_Music_Detection_Library\_variant_scratch"     # transient
OUTPUT_CSV  = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\Human_augmented.csv"
LABEL       = 0            # 0 = human, 1 = AI  (this script builds the human half)
NUM_WORKERS = 4           # parallel feature-extraction workers (CPU-bound)

# The 5 codec variants per track. (codec_name, ffmpeg_bitrate_or_None).
# "lossless" processes the FLAC master directly (no re-encode, no delete).
CODEC_VARIANTS = [
    ("lossless", None),
    ("mp3_128",  "128k"),
    ("mp3_192",  "192k"),
    ("mp3_320",  "320k"),
    ("ogg",      "192k"),
]

# extra columns appended after the standard feature columns
EXTRA_FIELDS = ["baseTrackId", "codec", "musicType"]
FIELDNAMES   = list(FIELD_MAP.values()) + EXTRA_FIELDS


# ── codec variant generation + feature extraction (runs in worker process) ─────
def _make_variant(master_path, base_id, codec, bitrate, scratch_dir):
    """Create the codec variant on disk. Returns (target_path, should_delete)."""
    if codec == "lossless":
        return master_path, False        # featurize the master itself; never delete

    ext = "ogg" if codec == "ogg" else "mp3"
    out = os.path.join(scratch_dir, f"{base_id}_{codec}.{ext}")
    encoder = "libvorbis" if codec == "ogg" else "libmp3lame"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", master_path,
        "-c:a", encoder, "-b:a", bitrate,
        "-map_metadata", "-1",           # strip tags so metadata can't leak
        out,
    ]
    subprocess.run(cmd, check=True)
    return out, True


def _variant_worker(master_path, base_id, codec, bitrate, label, music_type, scratch_dir):
    """Encode one variant, extract its features, clean up. Returns (ok, data, meta)."""
    target = None
    should_delete = False
    try:
        target, should_delete = _make_variant(master_path, base_id, codec, bitrate, scratch_dir)
        ok, data, err = _process_single_file(target)
        if not ok:
            return (False, None, f"{codec}: {err}")
        data["label"] = label            # _process_single_file hardcodes 0; force ours
        return (True, data, music_type)
    except Exception as e:
        return (False, None, f"{codec}: {e}")
    finally:
        if should_delete and target and os.path.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass


# ── the human master source: stream commercial-safe FLAC-capable records ───────
def iter_safe_records():
    """Yield (record, music_type) for commercial-safe, downloadable tracks, evenly
    across fetch.GENRES (~TARGET_TRACKS/len(GENRES) each). Stable popularity order
    means re-runs see the same tracks, so resume works."""
    per_genre = fetch.TARGET_TRACKS // len(fetch.GENRES)
    seen = set()
    for genre in fetch.GENRES:
        got = 0
        page = 0
        while got < per_genre and page < fetch.MAX_PAGES:
            params = {
                "client_id": fetch.CLIENT_ID, "format": "json",
                "limit": fetch.PAGE_SIZE, "offset": page * fetch.PAGE_SIZE,
                "include": "licenses", "tags": genre,
                "audioformat": fetch.DOWNLOAD_FORMAT,
                "audiodlformat": fetch.DOWNLOAD_FORMAT,
                "order": "popularity_total",
            }
            try:
                r = requests.get(fetch.BASE_URL, params=params, timeout=30)
                r.raise_for_status()
                recs = r.json().get("results", [])
            except Exception as e:
                print(f"  [{genre}] page {page} failed ({e}); pausing 10s...")
                time.sleep(10)
                continue
            if not recs:
                break
            for rec in recs:
                if got >= per_genre:
                    break
                tid = str(rec.get("id", ""))
                if tid in seen:
                    continue
                if not fetch.is_commercial_ok(rec.get("license_ccurl", "")):
                    continue
                if not rec.get("audiodownload_allowed", False):
                    continue
                seen.add(tid)
                got += 1
                yield rec, genre
            page += 1
            time.sleep(fetch.REQUEST_PAUSE)
        print(f"  [{genre}] queued {got} tracks")


# ── master download (main process) ─────────────────────────────────────────────
_prov_header_needed = not os.path.exists(fetch.PROV_CSV)


def ensure_master(rec, base_id, music_type=""):
    """Return path to the FLAC master, downloading + verifying + logging it if new.
    Returns None if the track has no lossless version or the download fails."""
    global _prov_header_needed
    path = os.path.join(MASTER_DIR, f"{base_id}.flac")
    if os.path.exists(path):
        return path

    url = rec.get("audiodownload", "")
    if not url:
        return None
    try:
        content = requests.get(url, timeout=120).content
    except Exception as e:
        print(f"  {base_id}: download failed ({e}); skipping.")
        return None
    if content[:4] != b"fLaC":
        print(f"  {base_id}: no lossless FLAC available; skipping.")
        return None

    with open(path, "wb") as f:
        f.write(content)
    fetch.append_provenance({
        "track_id": base_id,
        "artist":   rec.get("artist_name", ""),
        "title":    rec.get("name", ""),
        "license":  rec.get("license_ccurl", ""),
        "genre":    music_type,
        "source_url": url,
        "filename": f"{base_id}.flac",
    }, write_header=_prov_header_needed)
    _prov_header_needed = False
    return path


# ── CSV resume + row writing ───────────────────────────────────────────────────
def load_done_variants():
    """Set of (baseTrackId, codec) already in the output CSV."""
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row.get("baseTrackId", ""), row.get("codec", "")))
    return done


def _flush(results, buffer):
    """Map worker results into CSV rows and write them out."""
    for base_id, codec, ok, data, meta in results:
        if not ok:
            print(f"  skip {base_id} [{codec}]: {meta}")
            continue
        row = {FIELD_MAP.get(k, k): v for k, v in data.items()}
        row["baseTrackId"] = base_id
        row["codec"] = codec
        row["musicType"] = meta
        buffer.append(row)
    if buffer:
        write_batch(buffer, OUTPUT_CSV, FIELDNAMES)
        buffer.clear()


def _drain(pending, buffer, until):
    """Block until len(pending) <= until, collecting finished results."""
    while len(pending) > until:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        results = []
        for fut in done:
            base_id, codec = pending.pop(fut)
            ok, data, meta = fut.result()
            results.append((base_id, codec, ok, data, meta))
        _flush(results, buffer)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    if fetch.CLIENT_ID == "PASTE_YOUR_JAMENDO_CLIENT_ID_HERE":
        raise SystemExit("Set CLIENT_ID in fetch_human_tracks.py first.")
    os.makedirs(MASTER_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    done_variants = load_done_variants()
    print(f"Resuming: {len(done_variants)} (track, codec) rows already done.")
    songs = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        pending = {}
        buffer = []
        for rec, music_type in iter_safe_records():
            base_id = str(rec.get("id", ""))
            needed = [(c, b) for (c, b) in CODEC_VARIANTS
                      if (base_id, c) not in done_variants]
            if not needed:
                continue                       # this song is fully done

            master = ensure_master(rec, base_id, music_type)
            if master is None:
                continue

            for codec, bitrate in needed:
                fut = pool.submit(_variant_worker, master, base_id, codec, bitrate,
                                  LABEL, music_type, SCRATCH_DIR)
                pending[fut] = (base_id, codec)

            songs += 1
            if songs % 25 == 0:
                print(f"  queued {songs} songs; {len(pending)} variants in flight...")

            # backpressure: don't let the queue (or scratch dir) grow unbounded
            if len(pending) >= NUM_WORKERS * 4:
                _drain(pending, buffer, until=NUM_WORKERS * 2)

            time.sleep(fetch.DOWNLOAD_PAUSE)

        _drain(pending, buffer, until=0)       # finish the stragglers

    print(f"\nDone. Queued {songs} songs -> up to {songs * len(CODEC_VARIANTS)} "
          f"feature rows in {OUTPUT_CSV}")
    print(f"FLAC masters kept in {MASTER_DIR}")


if __name__ == "__main__":
    main()


# ==============================================================================
# AI HALF: reuse the SAME augmentation on your local AI files so both halves are
# codec-matched. No Jamendo download -- just point at a folder of AI tracks.
#
#   from build_dataset import augment_local_folder
#   augment_local_folder(r"G:\AI_Music_Detection_Library\AI_Music2",
#                        r"C:\...\AI_augmented.csv", label=1)
# ==============================================================================
def augment_local_folder(folder=None, output_csv=None, label=1, num_workers=NUM_WORKERS,
                         scratch_dir=SCRATCH_DIR, meta=None, files=None):
    """Codec-augment + featurize audio masters with the identical 5-codec variants.
    Used for the AI half (or any pre-downloaded set).

    folder: process every audio file in this folder, OR
    files:  process exactly this explicit list of paths (e.g. a balanced subset).
    meta:   optional dict {base_id: music_type} so AI rows carry their genre/type,
            letting you verify the AI half's balance matches the human half. If a
            base_id isn't in meta, the filename prefix before the first '_' is used
            (so files named  pop_0001.wav  ->  musicType 'pop')."""
    from process_folder import find_audio_files
    global OUTPUT_CSV
    OUTPUT_CSV = output_csv                     # so _flush/_drain target this CSV
    os.makedirs(scratch_dir, exist_ok=True)
    done_variants = load_done_variants()
    meta = meta or {}

    masters = files if files is not None else find_audio_files(folder)
    print(f"AI/local augmentation: {len(masters)} masters in {folder}; "
          f"{len(done_variants)} rows already done.")
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        pending, buffer = {}, []
        for master in masters:
            base_id = os.path.splitext(os.path.basename(master))[0]
            music_type = meta.get(base_id, base_id.split("_")[0])   # tag the type
            needed = [(c, b) for (c, b) in CODEC_VARIANTS
                      if (base_id, c) not in done_variants]
            for codec, bitrate in needed:
                fut = pool.submit(_variant_worker, master, base_id, codec, bitrate,
                                  label, music_type, scratch_dir)
                pending[fut] = (base_id, codec)
            if len(pending) >= num_workers * 4:
                _drain(pending, buffer, until=num_workers * 2)
        _drain(pending, buffer, until=0)
    print(f"Done. Features in {output_csv}")
