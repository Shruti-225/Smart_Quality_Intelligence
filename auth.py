
from pathlib import Path
import sqlite3
import hashlib
import hmac
import secrets

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "outputs" / "auth"
DB_PATH = DB_DIR / "users.db"


def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        310000
    ).hex()

    return password_hash, salt


def register_user(full_name, email, username, password):
    full_name = full_name.strip()
    email = email.strip().lower()
    username = username.strip()

    if not all([full_name, email, username, password]):
        return False, "All fields are required."

    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Please enter a valid email address."

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash, salt = hash_password(password)

    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users
                (full_name, email, username, password_hash, salt)
                VALUES (?, ?, ?, ?, ?)
                """,
                (full_name, email, username, password_hash, salt)
            )

        return True, "Account created successfully. Please log in."

    except sqlite3.IntegrityError:
        return False, "Email or username already exists."


def authenticate_user(identifier, password):
    identifier = identifier.strip()

    with get_connection() as conn:
        user = conn.execute(
            """
            SELECT id, full_name, email, username,
                   password_hash, salt, role
            FROM users
            WHERE email = ? OR username = ?
            """,
            (identifier, identifier)
        ).fetchone()

    if not user:
        return None

    stored_hash, salt = user[4], user[5]
    entered_hash, _ = hash_password(password, salt)

    if not hmac.compare_digest(stored_hash, entered_hash):
        return None

    return {
        "id": user[0],
        "full_name": user[1],
        "email": user[2],
        "username": user[3],
        "role": user[6]
    }
