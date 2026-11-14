"""Local username/password authentication, stored in the same SQLite file as the event database.

This is a simple access gate for a single-machine student project, not a hardened multi-user
auth system: there is no rate-limiting, email verification, or password reset flow. Passwords
are never stored in plain text - PBKDF2-HMAC-SHA256 with a random per-user salt.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")
MIN_PASSWORD_LENGTH = 8
PBKDF2_ITERATIONS = 200_000
GENERIC_LOGIN_ERROR = "Invalid username or password."

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,
    salt            TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_login      TEXT
);
"""


class AuthError(ValueError):
    """Invalid credentials or input. Message is safe to show directly to the user."""


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS).hex()


class AuthManager:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as conn:
                conn.executescript(SCHEMA)
        except (sqlite3.Error, OSError) as exc:
            log.exception("Auth database initialisation failed")
            raise AuthError(f"Could not initialise the accounts database: {exc}") from exc

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            log.error("Auth database error: %s", exc)
            raise AuthError("An internal error occurred. Please try again.") from exc
        finally:
            conn.close()

    def user_exists(self, username: str) -> bool:
        with self._conn() as conn:
            return conn.execute("SELECT 1 FROM users WHERE username=?", (username.strip(),)).fetchone() is not None

    def signup(self, username: str, password: str, confirm_password: str, display_name: str = "") -> str:
        """Create an account. Returns the stored display name. Raises AuthError on any problem."""
        username = (username or "").strip()
        if not USERNAME_RE.match(username):
            raise AuthError("Username must be 3-32 characters: letters, digits, or underscore only.")
        if password != confirm_password:
            raise AuthError("Passwords do not match.")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        if self.user_exists(username):
            raise AuthError("That username is already taken.")
        display_name = display_name.strip() or username
        salt = secrets.token_bytes(16)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash, display_name) VALUES (?,?,?,?)",
                (username, salt.hex(), _hash_password(password, salt), display_name))
        log.info("New account created: %s", username)
        return display_name

    def login(self, username: str, password: str) -> str:
        """Verify credentials. Returns display_name on success. Raises AuthError (generic message,
        to avoid revealing whether a username exists) on any failure."""
        username = (username or "").strip()
        if not username or not password:
            raise AuthError(GENERIC_LOGIN_ERROR)
        with self._conn() as conn:
            row = conn.execute("SELECT salt, password_hash, display_name FROM users WHERE username=?",
                               (username,)).fetchone()
        if row is None:
            raise AuthError(GENERIC_LOGIN_ERROR)
        salt_hex, stored_hash, display_name = row
        candidate = _hash_password(password, bytes.fromhex(salt_hex))
        if not hmac.compare_digest(candidate, stored_hash):
            raise AuthError(GENERIC_LOGIN_ERROR)
        with self._conn() as conn:
            conn.execute("UPDATE users SET last_login=datetime('now') WHERE username=?", (username,))
        return display_name
