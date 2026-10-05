"""
ingest_sonics.py
SONICS route for the AI half -- SELECTIVE download version.

Instead of pulling all ~49k synthetic songs (~150-250 GB) and using only 4% of
them, this:
    1. downloads ONLY fake_songs.csv (a few MB),
    2. picks a genre-balanced subset aligned to the human side's 20 music types,
    3. downloads ONLY those ~2000 files (~8-12 GB),
    4. runs them through the SAME codec augmentation + feature extraction as
       everything else, label=1, genre-tagged.

AUTH (no CLI needed): set your Hugging Face token in the SAME PowerShell window
before running:

    $env:HF_TOKEN = "paste_your_token_here"
    python CodeFiles\\ingest_sonics.py

(If you hit a 401/403, open https://huggingface.co/datasets/awsaf49/sonics while
logged in, accept the terms, then re-run.)

-------------------------------------------------------------------------------
VERIFY on first run: the printed metadata columns + mapped-type counts. Confirm
AUDIO_SUBDIR (the repo folder holding the mp3s) and FILENAME_COL, and extend
GENRE_MAP if lots of tracks land in "other".
-------------------------------------------------------------------------------
"""

import os
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

import fetch_human_tracks as fetch                 # the 20 music types
from build_dataset import augment_local_folder      # the shared augmenter

# ── config ────────────────────────────────────────────────────────────────────
REPO_ID       = "awsaf49/sonics"
LOCAL_DIR     = r"G:\AI_Music_Detection_Library\SONICS"   # where selected files land
AUDIO_SUBDIR  = "fake_songs"          # folder inside the repo holding the mp3s (VERIFY)
METADATA_NAME = "fake_songs.csv"
AI_CSV        = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\AI_augmented.csv"

# fake_songs.csv columns: id, filename, title, duration, algorithm, style,
# platform, genre, mood, label, split.
FILENAME_COL = "filename"    # column that resolves to the actual audio file
GENRE_COL    = "genre"       # real genre column -> folded into our 20 types
DURATION_COL = "duration"    # seconds; used to drop sub-60s tracks
MIN_DURATION = 60            # process_folder's floor
AUDIO_EXT    = ".mp3"        # SONICS fake songs are mp3
PER_TYPE     = 80            # 80 SONICS + 20 self-generated = 100/type (matches human)

MUSIC_TYPES = fetch.GENRES   # SAME 20 types as the human half

# Fold SONICS' free-text genres into our 20 types. Extend as you see real labels.
GENRE_MAP = {
    "electro": "electronic", "edm": "electronic", "house": "electronic",
    "techno": "electronic", "dance": "electronic",
    "rock": "rock", "metal": "metal", "punk": "punk",
    "pop": "pop", "jazz": "jazz", "blues": "blues", "classical": "classical",
    "orchestr": "soundtrack", "cinemat": "soundtrack", "soundtrack": "soundtrack",
    "hip": "hiphop", "rap": "hiphop", "trap": "hiphop",
    "folk": "folk", "acoustic": "folk", "country": "country",
    "reggae": "reggae", "soul": "soul", "funk": "funk", "r&b": "rnb", "rnb": "rnb",
    "ambient": "ambient", "world": "world", "latin": "latin", "experiment": "experimental",
}


def map_genre(raw):
    """Map a SONICS genre string to one of our 20 types, or 'other'."""
    if not isinstance(raw, str):
        return "other"
    low = raw.lower()
    for needle, mapped in GENRE_MAP.items():
        if needle in low:
            return mapped
    return "other"


def get_metadata_path():
    """Download just fake_songs.csv (tiny) if we don't already have it."""
    local = os.path.join(LOCAL_DIR, METADATA_NAME)
    if os.path.exists(local):
        return local
    print(f"Downloading {METADATA_NAME} (metadata only)...")
    return hf_hub_download(REPO_ID, METADATA_NAME, repo_type="dataset",
                           local_dir=LOCAL_DIR)


def fetch_one(fname):
    """Download a single audio file from the repo; return its local path or None."""
    repo_path = f"{AUDIO_SUBDIR}/{fname}"
    try:
        return hf_hub_download(REPO_ID, repo_path, repo_type="dataset",
                               local_dir=LOCAL_DIR)
    except Exception as e:
        print(f"  couldn't fetch {repo_path} ({e}); skipping.")
        return None


def main():
    try:
        meta_path = get_metadata_path()
    except Exception as e:
        sys.exit(f"Could not download metadata ({e}). Is HF_TOKEN set / terms accepted?")

    df = pd.read_csv(meta_path)
    print("SONICS metadata columns:", list(df.columns))   # sanity-check on first run
    for col in (FILENAME_COL, GENRE_COL, DURATION_COL):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not in metadata; fix the *_COL config above.")

    # drop tracks under the 60s floor so we don't download files we'd reject anyway
    before = len(df)
    df = df[pd.to_numeric(df[DURATION_COL], errors="coerce") >= MIN_DURATION].copy()
    print(f"Kept {len(df)}/{before} tracks >= {MIN_DURATION}s.")

    df["mtype"] = df[GENRE_COL].map(map_genre)
    print("Mapped-type counts:\n", df["mtype"].value_counts())

    # balanced selection: up to PER_TYPE per music type
    picks = [df[df["mtype"] == mtype].head(PER_TYPE) for mtype in MUSIC_TYPES]
    selected = pd.concat(picks) if picks else df.head(0)
    print(f"Selected {len(selected)} tracks; downloading only these...")

    # download ONLY the selected files, one at a time
    files, meta = [], {}
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        fname = str(row[FILENAME_COL])
        if not os.path.splitext(fname)[1]:
            fname += AUDIO_EXT
        path = fetch_one(fname)
        if path is None:
            continue
        files.append(path)
        # key meta by the filename STEM (the base_id augment derives from the path)
        meta[os.path.splitext(os.path.basename(fname))[0]] = row["mtype"]
        if i % 100 == 0:
            print(f"  downloaded {i}/{len(selected)}...")

    print(f"Got {len(files)} SONICS files on disk.")
    if not files:
        sys.exit("No SONICS audio downloaded. Check AUDIO_SUBDIR / FILENAME_COL / token.")

    # SAME augmentation + features as the human half, label=1, genre-tagged.
    augment_local_folder(output_csv=AI_CSV, label=1, files=files, meta=meta)
    print(f"AI features (SONICS) in {AI_CSV}")


if __name__ == "__main__":
    main()
