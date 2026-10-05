#Standalone Script

#Install Packages:
# pip install transformers soundfile (should be already complete)
# pip install torch --index-url https://download.pytorch.org/whl/cu121

#Then Run:
# python CodeFiles\generate_ai_tracks_local.py






"""
generate_ai_tracks_local.py
FREE local AI-music generation for the diversity slice of the AI half, using
MusicGen (Meta) via Hugging Face transformers -- runs on your RTX 4060, no API,
no cost. Adds a THIRD generator (MusicGen) alongside SONICS's Suno + Udio so the
detector learns "AI in general", not one model's fingerprint.

WHAT IT DOES:
  * generates PER_TYPE tracks for each of the 20 music types (same types as the
    human half), at varied lengths >= 60s, named "<type>_<n>.wav"
  * saves them as WAV masters in OUT_DIR
  * hands them to the SAME augment_local_folder (label=1) as everything else

ONE-TIME INSTALL (Windows, for your NVIDIA GPU):
    pip install transformers soundfile
    pip install torch --index-url https://download.pytorch.org/whl/cu121
  (the cu121 wheel gives CUDA support for the RTX 4060; first run downloads the
   MusicGen-medium model, ~3 GB.)

RUN:
    python CodeFiles\\generate_ai_tracks_local.py
  Generation uses the GPU; the feature-extraction phase afterwards uses the CPU.
  Don't run this at the same time as ingest_sonics.py / build_dataset.py.

RESUME: a track whose WAV already exists is skipped, so you can stop/restart.
"""

import os
import random

import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

import fetch_human_tracks as fetch                 # the 20 music types
from build_dataset import augment_local_folder      # the shared augmenter

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "facebook/musicgen-medium"   # medium (1.5B). If you hit CUDA OOM on
                                           # 8 GB, switch to "facebook/musicgen-small".
OUT_DIR     = r"G:\AI_Music_Detection_Library\AI_Generated_Masters"
AI_CSV      = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\AI_augmented.csv"

MUSIC_TYPES = fetch.GENRES                 # SAME 20 types as the human half
PER_TYPE    = 20                           # 20 x 20 = 400 diversity tracks
MIN_SECONDS = 60                           # process_folder's floor
MAX_SECONDS = 90                           # keep memory/time reasonable on 8 GB
TOKENS_PER_SEC = 50                        # MusicGen audio frame rate (~50 Hz)

# Prompt flavour so the 20 tracks per type aren't identical (variety helps).
PROMPT_MOODS  = ["upbeat", "melancholic", "energetic", "dreamy", "aggressive",
                 "chill", "cinematic", "warm", "dark", "uplifting"]
PROMPT_EXTRAS = ["full arrangement", "clear mix", "studio quality",
                 "rich instrumentation", "professional production"]


def build_prompt(music_type):
    return f"{random.choice(PROMPT_MOODS)} {music_type}, {random.choice(PROMPT_EXTRAS)}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU detected -- generation will be VERY slow on CPU.")
    print(f"Loading {MODEL_NAME} on {device}...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=dtype).to(device)
    sr = model.config.audio_encoder.sampling_rate     # MusicGen outputs 32 kHz

    os.makedirs(OUT_DIR, exist_ok=True)
    total = len(MUSIC_TYPES) * PER_TYPE
    made = 0

    for music_type in MUSIC_TYPES:
        for n in range(PER_TYPE):
            base_id = f"{music_type}_{n:04d}"        # prefix = musicType downstream
            out_path = os.path.join(OUT_DIR, f"{base_id}.wav")
            if os.path.exists(out_path):
                made += 1
                continue

            seconds = random.randint(MIN_SECONDS, MAX_SECONDS)   # varied lengths
            prompt = build_prompt(music_type)
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                audio = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=int(seconds * TOKENS_PER_SEC),
                )
            wav = audio[0, 0].cpu().numpy().astype("float32")     # mono
            sf.write(out_path, wav, sr)

            made += 1
            print(f"  [{made}/{total}] {base_id}  ({seconds}s)  \"{prompt}\"")

    # Free VRAM before the CPU-heavy feature phase.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nGeneration done. Codec-augmenting + extracting features...")
    augment_local_folder(folder=OUT_DIR, output_csv=AI_CSV, label=1)
    print(f"AI features (MusicGen, local) in {AI_CSV}")


if __name__ == "__main__":
    main()
