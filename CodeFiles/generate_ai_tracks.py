"""
generate_ai_tracks.py
SELF-GENERATE route for the AI half: create AI music with a commercially-licensed,
API-driven generator (Mubert or Stable Audio), GENRE-ALIGNED to the human side,
then feed the results through the SAME codec augmentation + feature extraction as
everything else.

WHY THIS EXISTS ALONGSIDE SONICS:
  SONICS gives you bulk Suno/Udio tracks, but only those two generators. This adds
  generator diversity (so the detector learns "AI-ness", not "Suno-ness") and gives
  you a slice of the AI half with spotless commercial licensing.

GENRE ALIGNMENT:
  It generates the SAME 20 music types as the human half (imported from
  fetch_human_tracks.GENRES), at matched per-type counts, and names each file
  "<type>_<n>.wav" so its musicType flows straight into the feature CSV.

LENGTHS:
  Randomized in [MIN_SECONDS, MAX_SECONDS], kept >= 60s so process_folder accepts
  them (its floor is 60s) and <= 600s (its ceiling).

OUTPUT:
  Lossless WAV masters in OUT_DIR (so the "lossless" codec variant is honestly
  lossless), then augment_local_folder() writes rows into AI_CSV with label=1.

-------------------------------------------------------------------------------
IMPORTANT - VERIFY THE API ADAPTER the first time you run:
  Exact endpoint URLs / field names for these services change. Each adapter below
  is marked  # VERIFY  where you should confirm against the live docs:
    Mubert:       https://mubert.com/api            (royalty-free, has genre param)
    Stable Audio: https://platform.stability.ai/docs/api-reference  (prompt + secs)
  The first successful call prints the raw response so you can adjust if needed.
-------------------------------------------------------------------------------
"""

import os
import sys
import time
import random

import requests

import fetch_human_tracks as fetch                 # reuse the 20 music types
from build_dataset import augment_local_folder      # reuse the shared augmenter

# ── config ────────────────────────────────────────────────────────────────────
GENERATOR = "stableaudio"     # "stableaudio" or "mubert"
API_KEY   = "PASTE_YOUR_API_KEY_HERE"

OUT_DIR   = r"G:\AI_Music_Detection_Library\AI_Generated_Masters"
AI_CSV    = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\AI_augmented.csv"

MUSIC_TYPES   = fetch.GENRES                 # SAME 20 types as the human half
PER_TYPE      = 20                            # tracks per type (20 x 20 = 400 top-up)
MIN_SECONDS   = 60                            # process_folder floor
MAX_SECONDS   = 240                           # keep well under the 600s ceiling
REQUEST_PAUSE = 1.0

# Prompt flavour so we don't send 20 identical prompts per type (variety helps the
# detector generalise). One is picked at random per track and combined with the type.
PROMPT_MOODS = ["upbeat", "melancholic", "energetic", "dreamy", "aggressive",
                "chill", "cinematic", "warm", "dark", "uplifting"]
PROMPT_EXTRAS = ["full arrangement", "clear mix", "studio quality",
                 "rich instrumentation", "professional production"]


def build_prompt(music_type):
    return f"{random.choice(PROMPT_MOODS)} {music_type}, {random.choice(PROMPT_EXTRAS)}"


# ── generator adapters (return WAV bytes) ─────────────────────────────────────
def generate_stableaudio(prompt, seconds, api_key):
    """Stability AI text-to-audio. VERIFY endpoint/params against the live docs."""
    url = "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio"  # VERIFY
    headers = {"authorization": f"Bearer {api_key}", "accept": "audio/*"}
    files = {"none": ""}                     # multipart form; Stability quirk
    data = {
        "prompt": prompt,                    # VERIFY field name
        "seconds_total": seconds,            # VERIFY field name
        "output_format": "wav",              # VERIFY: we NEED lossless wav
    }
    r = requests.post(url, headers=headers, files=files, data=data, timeout=180)
    r.raise_for_status()
    return r.content


def generate_mubert(prompt, seconds, api_key, music_type):
    """Mubert generation. VERIFY endpoint/params + response shape against the docs.
    Mubert is task-based (request -> poll -> download); adjust to current API."""
    url = "https://api.mubert.com/v2/records"     # VERIFY endpoint
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "prompt": prompt,                    # VERIFY: Mubert also supports genre/mood
        "genre": music_type,                 # native genre control (alignment!)
        "duration": seconds,
        "format": "wav",                     # VERIFY: request lossless
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    resp = r.json()
    # VERIFY: Mubert usually returns a URL (possibly after a short processing wait).
    audio_url = resp.get("data", {}).get("download_url") or resp.get("url")
    if not audio_url:
        raise RuntimeError(f"no audio url in response: {resp}")
    return requests.get(audio_url, timeout=180).content


def generate(prompt, seconds, music_type):
    if GENERATOR == "stableaudio":
        return generate_stableaudio(prompt, seconds, API_KEY)
    if GENERATOR == "mubert":
        return generate_mubert(prompt, seconds, API_KEY, music_type)
    raise SystemExit(f"unknown GENERATOR: {GENERATOR}")


# ── main ──────────────────────────────────────────────────────────────────────
def existing_ids():
    """WAV masters already generated -> resume without re-paying for them."""
    if not os.path.isdir(OUT_DIR):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(OUT_DIR)
            if f.lower().endswith(".wav")}


def main():
    if API_KEY == "PASTE_YOUR_API_KEY_HERE":
        sys.exit(f"Set API_KEY for {GENERATOR} first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    done = existing_ids()
    print(f"Generating with {GENERATOR}: {PER_TYPE} per type x {len(MUSIC_TYPES)} "
          f"types. {len(done)} already on disk.")
    printed_sample = False

    for music_type in MUSIC_TYPES:
        for n in range(PER_TYPE):
            base_id = f"{music_type}_{n:04d}"     # prefix = musicType downstream
            out_path = os.path.join(OUT_DIR, f"{base_id}.wav")
            if base_id in done or os.path.exists(out_path):
                continue

            seconds = random.randint(MIN_SECONDS, MAX_SECONDS)   # varied lengths
            prompt = build_prompt(music_type)
            try:
                audio = generate(prompt, seconds, music_type)
            except Exception as e:
                print(f"  {base_id}: generation failed ({e}); skipping.")
                time.sleep(REQUEST_PAUSE)
                continue

            if not printed_sample:
                print(f"  first call OK: {len(audio)} bytes for '{prompt}' ({seconds}s)")
                printed_sample = True

            with open(out_path, "wb") as f:
                f.write(audio)
            print(f"  generated {base_id} ({seconds}s)")
            time.sleep(REQUEST_PAUSE)

    # Feed the generated WAVs through the SAME augmentation as the human half.
    # Filenames are "<type>_<n>.wav", so augment_local_folder tags musicType from
    # the prefix automatically -> AI rows carry genre for balance-checking.
    print("\nGeneration done. Codec-augmenting + extracting features...")
    augment_local_folder(folder=OUT_DIR, output_csv=AI_CSV, label=1)
    print(f"AI features (self-generated) in {AI_CSV}")


if __name__ == "__main__":
    main()
