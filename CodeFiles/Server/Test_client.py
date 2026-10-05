# ── test_client.py ────────────────────────────────────────────────────────────
# Quick script to test the auth server while it's running.
# 1. In one terminal:  uvicorn Server:app --reload --port 8000
# 2. In another:        python test_client.py
#
# Requires: pip install requests

import requests

BASE = "http://localhost:8000"

def show(label, r):
    try:
        print(f"{label:20} {r.status_code}  {r.json()}")
    except Exception:
        print(f"{label:20} {r.status_code}  {r.text}")

if __name__ == "__main__":
    show("health", requests.get(f"{BASE}/health"))

    # Use a unique-ish email each run, or it'll 409 on the second run
    email = "tester@example.com"
    pw = "password123"

    r = requests.post(f"{BASE}/signup", json={"email": email, "password": pw})
    show("signup", r)

    if r.status_code == 409:
        # already exists from a prior run — just log in
        r = requests.post(f"{BASE}/login", json={"email": email, "password": pw})
        show("login", r)

    token = r.json().get("token")
    show("me", requests.get(f"{BASE}/me",
                            headers={"Authorization": f"Bearer {token}"}))