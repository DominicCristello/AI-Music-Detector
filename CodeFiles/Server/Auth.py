# ── auth.py ───────────────────────────────────────────────────────────────────
# Password hashing (bcrypt) + session tokens (JWT). SERVER-SIDE ONLY.

import bcrypt
import jwt
import datetime
from Config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_HOURS


# ── Password hashing ──────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt. Returns a string to store in DB."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode('utf-8'), salt).decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


# ── JWT session tokens ────────────────────────────────────────────────────────
def create_token(email: str, token_version: int = 0) -> str:
    """
    Issue a signed JWT for a logged-in user.

    `token_version` is embedded so a password reset (which bumps the stored
    version) invalidates every token issued before it.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": email.strip().lower(),                       # subject = the user
        "ver": token_version,                               # session generation
        "iat": now,                                         # issued at
        "exp": now + datetime.timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str):
    """
    Validate a JWT. Returns the decoded payload dict (with "sub" and "ver")
    if valid, else None. Handles expired and tampered tokens. The caller is
    responsible for checking "ver" against the user's current token_version.
    """
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None