from database.auth_manager import AuthError, AuthManager, MIN_PASSWORD_LENGTH


def _auth(tmp_path):
    return AuthManager(tmp_path / "auth" / "test.db")


def test_signup_and_login_roundtrip(tmp_path):
    a = _auth(tmp_path)
    name = a.signup("alice01", "correcthorse1", "correcthorse1", "Alice")
    assert name == "Alice"
    assert a.user_exists("alice01")
    assert a.login("alice01", "correcthorse1") == "Alice"


def test_signup_without_display_name_falls_back_to_username(tmp_path):
    a = _auth(tmp_path)
    assert a.signup("bob_02", "longenoughpw", "longenoughpw") == "bob_02"


def test_wrong_password_rejected_with_generic_message(tmp_path):
    a = _auth(tmp_path)
    a.signup("carol", "realpassword1", "realpassword1")
    try:
        a.login("carol", "wrongpassword")
        assert False, "should have raised"
    except AuthError as exc:
        assert "Invalid username or password" in str(exc)


def test_unknown_user_gets_same_generic_message(tmp_path):
    a = _auth(tmp_path)
    try:
        a.login("ghost", "whatever123")
        assert False
    except AuthError as exc:
        assert "Invalid username or password" in str(exc)


def test_duplicate_username_rejected(tmp_path):
    a = _auth(tmp_path)
    a.signup("dupe", "password123", "password123")
    try:
        a.signup("dupe", "password456", "password456")
        assert False
    except AuthError as exc:
        assert "already taken" in str(exc)


def test_password_mismatch_rejected(tmp_path):
    a = _auth(tmp_path)
    try:
        a.signup("dana", "password123", "password124")
        assert False
    except AuthError as exc:
        assert "do not match" in str(exc)


def test_short_password_rejected(tmp_path):
    a = _auth(tmp_path)
    try:
        a.signup("eve", "short", "short")
        assert False
    except AuthError as exc:
        assert str(MIN_PASSWORD_LENGTH) in str(exc)


def test_invalid_username_rejected(tmp_path):
    a = _auth(tmp_path)
    for bad in ("ab", "has space", "semi;colon", "x" * 40):
        try:
            a.signup(bad, "password123", "password123")
            assert False, bad
        except AuthError:
            pass


def test_password_is_never_stored_in_plain_text(tmp_path):
    import sqlite3
    a = _auth(tmp_path)
    a.signup("frank", "mysecretpassword", "mysecretpassword")
    with sqlite3.connect(a.path) as conn:
        row = conn.execute("SELECT password_hash, salt FROM users WHERE username='frank'").fetchone()
    assert "mysecretpassword" not in row[0]
    assert row[0] != "mysecretpassword" and len(row[0]) == 64  # sha256 hex digest length


def test_login_updates_last_login(tmp_path):
    import sqlite3
    a = _auth(tmp_path)
    a.signup("grace", "password123", "password123")
    a.login("grace", "password123")
    with sqlite3.connect(a.path) as conn:
        last = conn.execute("SELECT last_login FROM users WHERE username='grace'").fetchone()[0]
    assert last is not None
