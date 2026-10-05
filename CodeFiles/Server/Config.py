# ── config.py ─────────────────────────────────────────────────────────────────
# Server-side configuration. THIS FILE LIVES ON YOUR SERVER ONLY.
# It holds secrets (Stripe secret key, JWT signing secret) and must NEVER be
# bundled into the downloadable desktop app.
#
# For real deployment, read these from environment variables instead of
# hard-coding them (see os.environ usage below) so secrets never sit in source.

import os
import sys
import importlib.util

# ── Stripe ────────────────────────────────────────────────────────────────────
# Use TEST keys (sk_test_...) while developing. Switch to live keys only in
# production. Get these from https://dashboard.stripe.com/test/apikeys
STRIPE_SECRET_KEY = os.environ.get(
    "STRIPE_SECRET_KEY",
    "sk_test_REPLACE_ME"          # ← your Stripe TEST secret key
)

# Map each Stripe Price ID → the plan/tier name your app understands.
# Create products+prices in the Stripe dashboard, then paste their price IDs.
# The values here ('Basic'/'Premium'/'Business') are translated to your app's
# tier constants in the /me endpoint.
STRIPE_PRICE_IDS = {
    "price_REPLACE_BASIC":    "Basic",
    "price_REPLACE_PREMIUM":  "Premium",
    "price_REPLACE_BUSINESS": "Business",
}

# Map the human plan name → the tier string the desktop app expects.
PLAN_TO_TIER = {
    "Basic":    "basic_model",
    "Premium":  "premium_model",
    "Business": "business_model",
}

# Default tier for a logged-in user with NO active paid subscription.
# Basic is your free default tier, so new users land here.
DEFAULT_TIER = "basic_model"

# ── JWT (session tokens) ──────────────────────────────────────────────────────
# This secret signs the login tokens. If it leaks, anyone can forge logins —
# keep it server-side only and use a long random value in production.
JWT_SECRET = os.environ.get(
    "JWT_SECRET",
    "CHANGE_ME_to_a_long_random_string_in_production"
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7        # token valid for 7 days

# ── Email (verification + password-reset codes) ───────────────────────────────
# To send REAL emails, put your Gmail address + App Password in email_secrets.py
# (sits next to this file). See that file for the 3-step Gmail setup. You can
# also use environment variables instead — those take priority.
#
# If EMAIL_ENABLED is False (nothing configured), codes are printed to
# server.log instead of emailed — handy for testing before configuring SMTP.

# Optional private overrides from email_secrets.py (never committed/shared).
#
# Source runs import the ignored file beside Config.py.  A frozen server must
# not embed that file because doing so would place the Gmail app password in a
# redistributable EXE, so personal builds keep an external private copy in
# LocalAppData instead.
def _load_email_secrets():
    try:
        import email_secrets
        return email_secrets
    except Exception:
        pass

    if getattr(sys, 'frozen', False):
        private_path = os.path.join(
            os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
            'AI Music Detector', 'email_secrets.py')
        if os.path.isfile(private_path):
            try:
                spec = importlib.util.spec_from_file_location(
                    '_ai_detector_email_secrets', private_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            except Exception as exc:
                print(f'[email] Could not load private email settings: {exc}')
    return None


_secrets = _load_email_secrets()

def _email_cfg(name, default=""):
    """Priority: environment variable > email_secrets.py > default."""
    if os.environ.get(name):
        return os.environ[name]
    if _secrets is not None and getattr(_secrets, name, ""):
        return getattr(_secrets, name)
    return default

SMTP_HOST      = _email_cfg("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(_email_cfg("SMTP_PORT", "587"))
SMTP_USER      = _email_cfg("SMTP_USER", "").strip()          # your gmail address
SMTP_PASSWORD  = _email_cfg("SMTP_PASSWORD", "").replace(" ", "")  # app password
SMTP_FROM_NAME = _email_cfg("SMTP_FROM_NAME", "AI Music Detector")
CONTACT_EMAIL  = _email_cfg("CONTACT_EMAIL", "cristellodominic@gmail.com").strip()

# Auto-enable real sending only when both user + password are set.
EMAIL_ENABLED  = bool(SMTP_USER and SMTP_PASSWORD)

# How long a reset code is valid, and how many verify attempts it allows.
RESET_CODE_TTL_MINUTES = 10
RESET_CODE_MAX_ATTEMPTS = 5

# Signup email-verification codes expire faster than reset codes.
SIGNUP_CODE_TTL_MINUTES = 2

# ── Database ──────────────────────────────────────────────────────────────────
# Source runs keep the existing development database. A packaged one-file
# server is temporary, so its persistent database belongs in LocalAppData.
if getattr(sys, 'frozen', False):
    _DATA_DIR = os.path.join(
        os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
        'AI Music Detector')
    os.makedirs(_DATA_DIR, exist_ok=True)
    DB_PATH = os.path.join(_DATA_DIR, 'users.db')

else:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
