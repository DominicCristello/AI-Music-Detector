# ── db.py ─────────────────────────────────────────────────────────────────────
# SQLite user store. SERVER-SIDE ONLY.
# Table: users(id, email, password_hash, stripe_customer_id, created_at)

import sqlite3
import datetime
from Config import DB_PATH


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row      # rows behave like dicts
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every boot."""
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                email             TEXT UNIQUE NOT NULL,
                password_hash     TEXT NOT NULL,
                stripe_customer_id TEXT,
                created_at        TEXT NOT NULL,
                token_version     INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Migrate DBs created before token_version existed.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)")]
        if "token_version" not in cols:
            conn.execute(
                "ALTER TABLE users ADD COLUMN "
                "token_version INTEGER NOT NULL DEFAULT 0")
        # Pending password-reset codes. One active row per email (we delete old
        # rows for that email when a new code is requested).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reset_codes (
                email       TEXT PRIMARY KEY,
                code        TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                attempts    INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Pending (unverified) signups. The account is only written to `users`
        # once the emailed verification code is confirmed. One row per email.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_signups (
                email         TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                code          TEXT NOT NULL,
                expires_at    TEXT NOT NULL,
                attempts      INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
    finally:
        conn.close()


def create_user(email: str, password_hash: str, stripe_customer_id: str):
    """Insert a new user. Returns (success, error). Email must be unique."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, stripe_customer_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (email.strip().lower(), password_hash, stripe_customer_id,
             datetime.datetime.utcnow().isoformat())
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "An account with that email already exists."
    except Exception as e:
        return False, f"Database error: {e}"
    finally:
        conn.close()


def get_user_by_email(email: str):
    """Return the user row as a dict, or None if not found."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_password(email: str, new_password_hash: str) -> bool:
    """
    Set a new password hash for a user AND bump token_version in the same
    transaction, so every JWT issued before the reset stops validating.
    Returns True if a row was updated.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE users SET password_hash = ?, "
            "token_version = token_version + 1 WHERE email = ?",
            (new_password_hash, email.strip().lower())
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Password-reset codes ──────────────────────────────────────────────────────
def save_reset_code(email: str, code: str, expires_at_iso: str):
    """Store (or replace) a reset code for an email, resetting attempts to 0."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO reset_codes (email, code, expires_at, attempts) "
            "VALUES (?, ?, ?, 0) "
            "ON CONFLICT(email) DO UPDATE SET "
            "code=excluded.code, expires_at=excluded.expires_at, attempts=0",
            (email.strip().lower(), code, expires_at_iso)
        )
        conn.commit()
    finally:
        conn.close()


def get_reset_code(email: str):
    """Return the reset-code row as a dict, or None."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM reset_codes WHERE email = ?",
            (email.strip().lower(),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def increment_reset_attempts(email: str):
    """Bump the attempts counter for a pending reset code."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE reset_codes SET attempts = attempts + 1 WHERE email = ?",
            (email.strip().lower(),)
        )
        conn.commit()
    finally:
        conn.close()


def delete_reset_code(email: str):
    """Remove a reset code (after success or when it's invalid)."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM reset_codes WHERE email = ?",
                     (email.strip().lower(),))
        conn.commit()
    finally:
        conn.close()


# ── Pending (unverified) signups ──────────────────────────────────────────────
def save_pending_signup(email: str, password_hash: str, code: str,
                        expires_at_iso: str):
    """Store (or replace) a pending signup + verification code, attempts -> 0."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO pending_signups "
            "(email, password_hash, code, expires_at, attempts) "
            "VALUES (?, ?, ?, ?, 0) "
            "ON CONFLICT(email) DO UPDATE SET "
            "password_hash=excluded.password_hash, code=excluded.code, "
            "expires_at=excluded.expires_at, attempts=0",
            (email.strip().lower(), password_hash, code, expires_at_iso)
        )
        conn.commit()
    finally:
        conn.close()


def get_pending_signup(email: str):
    """Return the pending-signup row as a dict, or None."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM pending_signups WHERE email = ?",
            (email.strip().lower(),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def increment_pending_attempts(email: str):
    """Bump the attempts counter for a pending signup's verification code."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE pending_signups SET attempts = attempts + 1 WHERE email = ?",
            (email.strip().lower(),)
        )
        conn.commit()
    finally:
        conn.close()


def delete_pending_signup(email: str):
    """Remove a pending signup (after verification succeeds or it expires)."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM pending_signups WHERE email = ?",
                     (email.strip().lower(),))
        conn.commit()
    finally:
        conn.close()