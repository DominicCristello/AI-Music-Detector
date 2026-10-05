# ── server.py ─────────────────────────────────────────────────────────────────
# FastAPI auth server for the AI Music Detector.  SERVER-SIDE ONLY.
#
# Run (development):
#     pip install fastapi "uvicorn[standard]" pyjwt bcrypt "pydantic[email]" stripe
#     uvicorn Server:app --reload --port 8000
#
# Then open http://localhost:8000/docs for interactive API docs.
#
# Endpoints:
#   POST /signup         {email, password}  → emails a 6-digit code, holds a pending signup
#   POST /signup/verify  {email, code}      → confirms code, creates user + Stripe customer, returns token+tier
#   POST /login          {email, password}  → verifies password, returns token+tier
#   GET  /me      (Authorization: Bearer <token>) → returns the current user's tier
#   GET  /health                           → simple liveness check

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
import secrets
import datetime
from contextlib import asynccontextmanager

import Db as db
import Auth as auth
from Config import (PLAN_TO_TIER, DEFAULT_TIER,
                    RESET_CODE_TTL_MINUTES, RESET_CODE_MAX_ATTEMPTS,
                    SIGNUP_CODE_TTL_MINUTES, CONTACT_EMAIL)
from Email_utils import send_email, send_reset_code, send_signup_code

# Stripe is optional during early development. If the key isn't set (still the
# placeholder), signup still works (without creating a real Stripe customer) so
# you can test auth. Auth-only mode kicks in when EITHER the stripe package is
# missing OR the secret key hasn't been configured yet.
from Config import STRIPE_SECRET_KEY

_STRIPE_KEY_CONFIGURED = bool(STRIPE_SECRET_KEY) and "REPLACE" not in STRIPE_SECRET_KEY

if _STRIPE_KEY_CONFIGURED:
    try:
        from Stripe_utils import create_stripe_customer, get_plan
        _STRIPE_READY = True
    except Exception as e:
        print(f"[server] Stripe package not available ({e}); running in auth-only mode.")
        _STRIPE_READY = False
else:
    print("[server] Stripe key not configured; running in auth-only mode.")
    _STRIPE_READY = False

if not _STRIPE_READY:
    def create_stripe_customer(email, name=""): return True, None, None
    def get_plan(cid): return None


@asynccontextmanager
async def lifespan(app):
    db.init_db()       # create tables / run migrations on startup
    yield

app = FastAPI(title="AI Music Detector — Auth API", lifespan=lifespan)

# Allow the /docs page (and any browser front-end) to call the API. The real
# client is a desktop app, so CORS — a browser mechanism — is mostly moot here;
# tighten allow_origins if a web front-end is ever added.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/response models ───────────────────────────────────────────────────
class Credentials(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    token: str
    tier: str
    email: str

class ForgotRequest(BaseModel):
    email: EmailStr

class VerifyRequest(BaseModel):
    email: EmailStr
    code: str

class ResetRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: str

class ContactRequest(BaseModel):
    title: str
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_tier(stripe_customer_id: Optional[str]) -> str:
    """Ask Stripe for the active plan and translate it to an app tier."""
    if not stripe_customer_id:
        return DEFAULT_TIER
    plan = get_plan(stripe_customer_id)          # 'Basic'/'Premium'/'Business'/None
    if not plan:
        return DEFAULT_TIER
    return PLAN_TO_TIER.get(plan, DEFAULT_TIER)


def _validate_password(password: str):
    """
    Enforce password length bounds. The upper bound matters: bcrypt silently
    ignores everything past 72 bytes, so without it two long passwords sharing
    their first 72 bytes would verify as equal.
    """
    if len(password) < 8:
        raise HTTPException(status_code=400,
                            detail="Password must be at least 8 characters.")
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400,
                            detail="Password must be at most 72 bytes.")


def _current_email(authorization: Optional[str]) -> str:
    """Extract + validate the bearer token, returning the user's email."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token.")
    token = authorization.split(" ", 1)[1].strip()
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token expired or invalid.")
    email = payload.get("sub")
    user = db.get_user_by_email(email) if email else None
    # Reject tokens whose version is stale (e.g. issued before a password reset).
    if not user or payload.get("ver") != user.get("token_version", 0):
        raise HTTPException(status_code=401,
                            detail="Session no longer valid. Please log in again.")
    return email


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "stripe": _STRIPE_READY}


@app.post("/signup")
def signup(creds: Credentials):
    """
    Step 1 of signup: validate, then email a 6-digit verification code and hold
    the account as *pending*. The user is NOT created until /signup/verify
    confirms the code — this proves they actually own the email address.
    """
    email = creds.email.strip().lower()
    _validate_password(creds.password)
    if db.get_user_by_email(email):
        raise HTTPException(status_code=409,
                            detail="An account with that email already exists.")

    # Stash the (hashed) password + a fresh code as a pending signup, then email
    # the code. No user row and no Stripe customer are created until it's
    # verified, so unverified emails leave nothing behind.
    pw_hash = auth.hash_password(creds.password)
    code = _gen_code()
    expires = (datetime.datetime.utcnow()
               + datetime.timedelta(minutes=SIGNUP_CODE_TTL_MINUTES))
    db.save_pending_signup(email, pw_hash, code, expires.isoformat())
    sent, _email_error = send_signup_code(email, code)
    if not sent:
        # Do not leave a pending account whose verification code the user can
        # never receive.  They can retry signup once mail delivery is restored.
        db.delete_pending_signup(email)
        raise HTTPException(
            status_code=503,
            detail="The verification email could not be delivered. "
                   "Please retry shortly.")
    return {"message": "A verification code has been sent to your email.",
            "email": email}


@app.post("/signup/verify", response_model=AuthResponse)
def signup_verify(req: VerifyRequest):
    """
    Step 2 of signup: check the emailed code and, on success, actually create
    the account (+ Stripe customer) and return a session token.
    """
    email = _check_pending_code(req.email, req.code)
    pending = db.get_pending_signup(email)
    if not pending:
        raise HTTPException(status_code=400,
                            detail="No pending signup found. Please sign up again.")

    # Someone may have registered this email in the meantime.
    if db.get_user_by_email(email):
        db.delete_pending_signup(email)
        raise HTTPException(status_code=409,
                            detail="An account with that email already exists.")

    # Create a Stripe customer (no-op in auth-only mode)
    ok, customer_id, err = create_stripe_customer(email)
    if not ok:
        raise HTTPException(status_code=502, detail=f"Stripe error: {err}")

    created, db_err = db.create_user(email, pending["password_hash"], customer_id)
    if not created:
        db.delete_pending_signup(email)
        raise HTTPException(status_code=409, detail=db_err)

    db.delete_pending_signup(email)                     # one-time use
    token = auth.create_token(email, token_version=0)   # brand-new account
    tier = _resolve_tier(customer_id)
    return AuthResponse(token=token, tier=tier, email=email)


@app.post("/login", response_model=AuthResponse)
def login(creds: Credentials):
    email = creds.email.strip().lower()
    user = db.get_user_by_email(email)
    if not user or not auth.verify_password(creds.password, user["password_hash"]):
        # Same message for both cases — don't leak which emails exist.
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = auth.create_token(email, user.get("token_version", 0))
    tier = _resolve_tier(user.get("stripe_customer_id"))
    return AuthResponse(token=token, tier=tier, email=email)


@app.get("/me")
def me(authorization: Optional[str] = Header(None)):
    email = _current_email(authorization)
    user = db.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    tier = _resolve_tier(user.get("stripe_customer_id"))
    return {"email": email, "tier": tier}


@app.post("/contact")
def contact(req: ContactRequest,
            authorization: Optional[str] = Header(None)):
    """Relay a contact message from a verified, signed-in desktop account."""
    sender_email = _current_email(authorization)
    title = req.title.strip()
    message = req.message.strip()

    if not title or not message:
        raise HTTPException(
            status_code=422,
            detail="Title and message must contain non-whitespace data.")
    if len(req.title) > 100:
        raise HTTPException(status_code=422,
                            detail="Title exceeds the 100-character limit.")
    if len(req.message) > 1000:
        raise HTTPException(status_code=422,
                            detail="Message exceeds the 1000-character limit.")
    if '\r' in title or '\n' in title:
        raise HTTPException(status_code=422,
                            detail="Title must be a single line.")

    body = (
        "Authenticated account contact request\n"
        f"From account: {sender_email}\n\n"
        f"{message}"
    )
    sent, _error = send_email(
        CONTACT_EMAIL,
        f"[AI Music Detector Contact] {title}",
        body,
        reply_to=sender_email)
    if not sent:
        raise HTTPException(
            status_code=503,
            detail="The mail transport rejected the message. Please retry later.")
    return {"message": "Contact message transmitted successfully."}


# ── Password reset flow ───────────────────────────────────────────────────────
def _gen_code() -> str:
    """Cryptographically-secure random 6-digit numeric code."""
    return f"{secrets.randbelow(1_000_000):06d}"


@app.post("/forgot-password")
def forgot_password(req: ForgotRequest):
    """
    Step 1: user requests a reset code.
    SECURITY: always returns success, even if the email has no account — this
    prevents using the endpoint to discover which emails are registered.
    A code is only generated/emailed if the account actually exists.
    """
    email = req.email.strip().lower()
    user = db.get_user_by_email(email)
    if user:
        code = _gen_code()
        expires = (datetime.datetime.utcnow()
                   + datetime.timedelta(minutes=RESET_CODE_TTL_MINUTES))
        db.save_reset_code(email, code, expires.isoformat())
        send_reset_code(email, code)      # emails it (or prints in dev mode)
    # Same response regardless of whether the account exists
    return {"message": "If that email has an account, a code has been sent."}


def _check_code(email: str, code: str) -> str:
    """
    Shared validation for verify + reset. Returns the lowercased email on
    success, raises HTTPException on any failure (expired / wrong / too many).
    """
    email = email.strip().lower()
    row = db.get_reset_code(email)
    if not row:
        raise HTTPException(status_code=400,
                            detail="No reset request found. Start again.")

    # Expired?
    try:
        expires = datetime.datetime.fromisoformat(row["expires_at"])
    except Exception:
        expires = datetime.datetime.utcnow()
    if datetime.datetime.utcnow() > expires:
        db.delete_reset_code(email)
        raise HTTPException(status_code=400,
                            detail="Code expired. Request a new one.")

    # Too many attempts?
    if row["attempts"] >= RESET_CODE_MAX_ATTEMPTS:
        db.delete_reset_code(email)
        raise HTTPException(status_code=429,
                            detail="Too many attempts. Request a new code.")

    # Wrong code? (constant-time compare to avoid leaking it via timing)
    if not secrets.compare_digest(code.strip(), row["code"]):
        db.increment_reset_attempts(email)
        raise HTTPException(status_code=400, detail="Incorrect code.")

    return email


def _check_pending_code(email: str, code: str) -> str:
    """
    Validate a pending-signup verification code. Returns the lowercased email on
    success, raises HTTPException on any failure (expired / wrong / too many).
    Mirrors _check_code but against the pending_signups table.
    """
    email = email.strip().lower()
    row = db.get_pending_signup(email)
    if not row:
        raise HTTPException(status_code=400,
                            detail="No pending signup found. Please sign up again.")

    # Expired?
    try:
        expires = datetime.datetime.fromisoformat(row["expires_at"])
    except Exception:
        expires = datetime.datetime.utcnow()
    if datetime.datetime.utcnow() > expires:
        db.delete_pending_signup(email)
        raise HTTPException(status_code=400,
                            detail="Code expired. Please sign up again.")

    # Too many attempts?
    if row["attempts"] >= RESET_CODE_MAX_ATTEMPTS:
        db.delete_pending_signup(email)
        raise HTTPException(status_code=429,
                            detail="Too many attempts. Please sign up again.")

    # Wrong code? (constant-time compare to avoid leaking it via timing)
    if not secrets.compare_digest(code.strip(), row["code"]):
        db.increment_pending_attempts(email)
        raise HTTPException(status_code=400, detail="Incorrect code.")

    return email


@app.post("/verify-code")
def verify_code(req: VerifyRequest):
    """
    Step 2: check the 6-digit code without consuming it (so the client can
    then show the new-password field, and /reset-password re-checks it).
    """
    _check_code(req.email, req.code)
    return {"message": "Code verified."}


@app.post("/reset-password")
def reset_password(req: ResetRequest):
    """
    Step 3: with a valid code, set the new password. Consumes the code.
    """
    _validate_password(req.new_password)
    email = _check_code(req.email, req.code)
    new_hash = auth.hash_password(req.new_password)
    if not db.update_password(email, new_hash):
        raise HTTPException(status_code=404, detail="User not found.")
    db.delete_reset_code(email)            # one-time use
    return {"message": "Password updated. You can now log in."}
