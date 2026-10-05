"""
fetch_human_tracks.py
Bulk-download commercially-usable (CC-BY / CC-BY-SA / CC0) FULL-LENGTH human-made
tracks from the Jamendo API, GENRE-BALANCED, for the "human" half of the
AI-music detector dataset.

WHY JAMENDO (and not FMA):
  The old Free Music Archive public API is SHUT DOWN, and FMA's static dataset
  only ships 30-SECOND clips unless you grab the ~879 GB full archive. Jamendo's
  API is alive and returns FULL-LENGTH downloads, filterable by CC license and
  by genre tag -- so it's the practical source for a varied full-song dataset.

ONE-TIME SETUP:
  Register a FREE app at https://developer.jamendo.com/v3.0/apps to get a
  client_id, then paste it into CLIENT_ID below. (No approval wait, no cost.)

WHAT THIS DOES, in one unattended run:
  * fetches a roughly EQUAL number of tracks per genre in GENRES (variety!)
  * keeps only tracks whose license permits COMMERCIAL use (CC-BY / BY-SA / CC0)
    and whose audio download is allowed
  * downloads the FULL-LENGTH audio into OUT_DIR
  * writes provenance.csv AS IT GOES (id, artist, title, license, genre, url)
    -> this file is BOTH your legal paper trail AND your CC-BY attribution list
  * tops up with an untagged pass so you still hit TARGET_TRACKS if a genre is thin
  * rate-limits so Jamendo doesn't throttle you
  * resumes cleanly: re-run after any interruption and it skips what it has
"""

import csv
import os
import time
import sys
import requests

# ── config ────────────────────────────────────────────────────────────────────
CLIENT_ID = "5d788e95"
BASE_URL  = "https://api.jamendo.com/v3.0/tracks/"
OUT_DIR   = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\human_audio"
PROV_CSV  = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\provenance.csv"

CENSUS_ONLY    = False    # True = NO downloads, just report per-type availability.
                          # False = DOWNLOAD. Census confirmed every type can supply
                          # 100 CC-safe tracks, so 100 x 20 types = 2000, evenly split.

# Audio format to download. "flac" = LOSSLESS masters (recommended): full quality,
# ~half the size of WAV, and transcodable to any canonical WAV spec later without
# quality loss. Other options: "mp32" (~192kbps MP3), "mp31" (96kbps), "ogg".
# We keep FLAC as the archival master and standardize to WAV as a SEPARATE later
# step, once the AI-generated half's format is locked (to avoid codec/samplerate
# leakage between the human and AI classes).
DOWNLOAD_FORMAT = "flac"
FILE_EXT        = {"flac": "flac", "mp32": "mp3", "mp31": "mp3", "ogg": "ogg"}[DOWNLOAD_FORMAT]
FLAC_ONLY_MASTERS = True  # if True, verify each download really is FLAC and skip any
                          # track that has no lossless version (keeps masters pure)

TARGET_TRACKS  = 2000     # total commercial-safe tracks you want (download mode)
PAGE_SIZE      = 200      # Jamendo hard max per request
REQUEST_PAUSE  = 1.0      # seconds between API pages (be polite)
DOWNLOAD_PAUSE = 0.5      # seconds between audio downloads
MAX_PAGES      = 500      # per-type safety cap so a bad loop can't run forever
CENSUS_SCAN_PAGES = 10    # pages (x200) to scan per type when counting CC-safe tracks

# MUSIC TYPES to spread the dataset across -- deliberately broad, not just "genres".
# Songs overlap genres, so we cast a wide net of *types* to maximize the breadth of
# sound the detector sees. These are Jamendo tag names; add/remove freely.
GENRES = [
    "electronic",
    "rock",
    "pop",
    "jazz",
    "blues",
    "metal",         # Jamendo tags heavy metal simply as "metal"
    "classical",
    "hiphop",
    "folk",
    "reggae",
    "country",
    "soul",
    "funk",
    "punk",
    "ambient",
    "soundtrack",    # cinematic / orchestral scores
    "world",         # non-Western / global styles
    "latin",
    "rnb",
    "experimental",
]

# Licenses that ALLOW commercial use are CC-BY and CC-BY-SA (and CC0/public
# domain). Anything NonCommercial ("nc") or NoDerivatives ("nd") is rejected.
# Jamendo reports the license as a URL like http://creativecommons.org/licenses/by/3.0/
LICENSE_BLOCK = ("nc", "nd")   # substrings in the license slug that disqualify


# ── helpers ───────────────────────────────────────────────────────────────────
def is_commercial_ok(license_url: str) -> bool:
    """True only if the CC license permits commercial use (BY / BY-SA / CC0)."""
    if not license_url:
        return False
    lu = license_url.lower()
    if "publicdomain" in lu or "/zero/" in lu or "cc0" in lu:
        return True
    if "/licenses/" not in lu:
        return False
    try:
        slug = lu.split("/licenses/")[1].split("/")[0]   # e.g. "by-nc-nd"
    except IndexError:
        return False
    parts = slug.split("-")
    if any(bad in parts for bad in LICENSE_BLOCK):
        return False
    return "by" in parts


def already_downloaded():
    """Track IDs already in provenance.csv -> lets the script resume."""
    done = set()
    if os.path.exists(PROV_CSV):
        with open(PROV_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                done.add(str(row['track_id']))
    return done


def append_provenance(row_dict, write_header):
    with open(PROV_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['track_id', 'artist', 'title',
                                          'license', 'genre', 'source_url',
                                          'filename'])
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


class State:
    """Shared counters/flags threaded through every fetch segment."""
    def __init__(self, done):
        self.done = done                 # set of track_ids we already have
        self.have = len(done)            # total tracks secured so far
        self.header_needed = not os.path.exists(PROV_CSV)
        self.printed_sample = False


def download_track(rec, genre_label, st):
    """Validate license, download the full audio, log provenance. Returns bool."""
    tid = str(rec.get("id", "")).strip()
    lic = rec.get("license_ccurl", "")
    if not tid or tid in st.done:
        return False
    if not is_commercial_ok(lic):
        return False
    if not rec.get("audiodownload_allowed", False):
        return False
    audio_url = rec.get("audiodownload", "")
    if not audio_url:
        return False

    fname = f"{tid}.{FILE_EXT}"
    fpath = os.path.join(OUT_DIR, fname)
    try:
        a = requests.get(audio_url, timeout=120)
        a.raise_for_status()
        content = a.content
    except Exception as e:
        print(f"  download failed for {tid} ({e}); skipping.")
        return False

    # Parity guard: if we asked for lossless masters, make sure this really IS a
    # FLAC (some tracks have no lossless version and Jamendo serves a fallback).
    if DOWNLOAD_FORMAT == "flac" and FLAC_ONLY_MASTERS and not content[:4] == b"fLaC":
        print(f"  {tid}: no lossless FLAC available; skipping to keep masters pure.")
        return False

    with open(fpath, 'wb') as out:
        out.write(content)

    append_provenance({
        'track_id':  tid,
        'artist':    rec.get("artist_name", ""),
        'title':     rec.get("name", ""),
        'license':   lic,
        'genre':     genre_label,
        'source_url': audio_url,
        'filename':  fname,
    }, write_header=st.header_needed)
    st.header_needed = False
    st.done.add(tid)
    st.have += 1
    return True


def fetch_segment(st, tag, stop_at, genre_label):
    """Page through Jamendo (optionally filtered by `tag`) until st.have hits
    stop_at or the tag runs out of tracks."""
    page = 0
    while st.have < stop_at and page < MAX_PAGES:
        params = {
            "client_id": CLIENT_ID,
            "format":    "json",
            "limit":     PAGE_SIZE,
            "offset":    page * PAGE_SIZE,
            "include":   "licenses",
            "audioformat":   DOWNLOAD_FORMAT,   # e.g. flac (lossless) full-length
            "audiodlformat": DOWNLOAD_FORMAT,
            "order":     "popularity_total",
        }
        if tag:
            params["tags"] = tag
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [{genre_label}] page {page} failed ({e}); pausing 10s...")
            time.sleep(10)
            continue

        records = data.get("results", [])
        if not records:
            break   # this tag is exhausted

        if not st.printed_sample:
            print("\n--- FIRST RAW RECORD (field names for reference) ---")
            print(records[0])
            print("--- end sample ---\n")
            st.printed_sample = True

        for rec in records:
            if st.have >= stop_at:
                break
            if download_track(rec, genre_label, st):
                if st.have % 25 == 0:
                    print(f"  {st.have}/{TARGET_TRACKS} tracks downloaded...")
                time.sleep(DOWNLOAD_PAUSE)

        page += 1
        time.sleep(REQUEST_PAUSE)


# ── census (no downloads) ─────────────────────────────────────────────────────
def census():
    """For each music type, report:
         raw  = total tracks Jamendo has for that tag (upper bound)
         safe = of the pages we scan, how many are commercial-safe + downloadable
                (a SAMPLE-based estimate of the usable pool, not the full catalog)
    Downloads nothing. Use the numbers to decide realistic per-type quotas."""
    print(f"Census over {len(GENRES)} music types "
          f"(scanning up to {CENSUS_SCAN_PAGES} pages x {PAGE_SIZE} each)...\n")
    print(f"  {'music type':<14} {'raw total':>10} {'CC-safe (sampled)':>20}")
    print(f"  {'-'*14} {'-'*10} {'-'*20}")
    results = {}
    for genre in GENRES:
        raw = None
        safe = 0
        scanned = 0
        for page in range(CENSUS_SCAN_PAGES):
            params = {
                "client_id": CLIENT_ID, "format": "json",
                "limit": PAGE_SIZE, "offset": page * PAGE_SIZE,
                "include": "licenses", "tags": genre,
                "order": "popularity_total",
            }
            try:
                r = requests.get(BASE_URL, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  {genre:<14}  request failed ({e})")
                break
            if raw is None:
                raw = data.get("headers", {}).get("results_fullcount", 0)
            recs = data.get("results", [])
            if not recs:
                break
            scanned += len(recs)
            for rec in recs:
                if (is_commercial_ok(rec.get("license_ccurl", ""))
                        and rec.get("audiodownload_allowed", False)):
                    safe += 1
            time.sleep(REQUEST_PAUSE)
        pct = (safe / scanned * 100) if scanned else 0
        results[genre] = {"raw": raw or 0, "safe": safe, "scanned": scanned}
        print(f"  {genre:<14} {raw or 0:>10} "
              f"{safe:>6} of {scanned:<5} sampled ({pct:.0f}% safe)")
    print("\nInterpretation: 'raw total' is everything tagged that type; 'CC-safe'")
    print("is how many of the top tracks we sampled are actually commercial-safe +")
    print("downloadable. Multiply the safe %% by the raw total for a rough usable")
    print("pool. Set TARGET_TRACKS and trim GENRES based on the thinnest types,")
    print("then flip CENSUS_ONLY = False to download.")
    return results


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if CLIENT_ID == "PASTE_YOUR_JAMENDO_CLIENT_ID_HERE":
        sys.exit("ERROR: register a free app at https://developer.jamendo.com/v3.0/apps "
                 "and set CLIENT_ID first.")

    if CENSUS_ONLY:
        census()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    st = State(already_downloaded())
    print(f"Resuming: {st.have} tracks already logged. Target: {TARGET_TRACKS} "
          f"across {len(GENRES)} genres.")

    per_genre = TARGET_TRACKS // len(GENRES)
    for i, genre in enumerate(GENRES, start=1):
        # cumulative goal: after genre i, we want ~ per_genre * i tracks total
        stop_at = min(TARGET_TRACKS, per_genre * i)
        print(f"\n=== Genre {i}/{len(GENRES)}: {genre} "
              f"(aiming for ~{stop_at} total so far) ===")
        fetch_segment(st, genre, stop_at, genre)

    # top-up: if some genres were thin, fill the rest with an untagged pass
    if st.have < TARGET_TRACKS:
        print(f"\n=== Top-up pass to reach {TARGET_TRACKS} "
              f"(currently {st.have}) ===")
        fetch_segment(st, None, TARGET_TRACKS, "mixed")

    print(f"\nDone. {st.have} commercial-safe full-length tracks in {OUT_DIR}")
    print(f"Attribution + provenance: {PROV_CSV}")
    print("Next: point process_folder at that folder to build human_data.csv, then retrain.")


if __name__ == "__main__":
    main()
