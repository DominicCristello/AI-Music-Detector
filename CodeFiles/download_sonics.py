#RUN: $env:HF_TOKEN = "paste_your_token_here"
# python CodeFiles\ingest_sonics.py

"""
download_sonics.py
Download ONLY the fake (AI-generated) half of the SONICS dataset from Hugging Face.

AUTH (no CLI needed): before running, set your token as an environment variable in
the SAME PowerShell window:

    $env:HF_TOKEN = "paste_your_token_here"
    python download_sonics.py

snapshot_download automatically uses HF_TOKEN. The token stays in this terminal
session only -- it is NOT written to any file.

(One-time gated-access note: if you get a 401/403, open
 https://huggingface.co/datasets/awsaf49/sonics while logged in, accept the terms,
 then re-run.)
"""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="awsaf49/sonics",
    repo_type="dataset",
    local_dir=r"G:\AI_Music_Detection_Library\SONICS",
    allow_patterns=["fake_songs/*", "fake_songs.csv"],   # skip the real-song half
)
print("Done. Check G:\\AI_Music_Detection_Library\\SONICS for fake_songs\\ + fake_songs.csv")
