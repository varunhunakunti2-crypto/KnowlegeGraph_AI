"""Tests for shared security helpers."""
import pytest

from app.core.security import (
    SecurityValidationError,
    clamp_int,
    is_loopback_host,
    is_path_within,
    redact_local_paths,
    safe_join_upload_path,
    sanitize_filename,
    validate_identifier,
    validate_uuid,
)


@pytest.mark.unit
def test_sanitize_filename_strips_traversal_and_paths():
    assert sanitize_filename("..\\..\\secret.pdf") == "secret.pdf"
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("..%2f..%2fevil.pdf") == "..%2f..%2fevil.pdf"


@pytest.mark.unit
def test_sanitize_filename_rejects_empty_or_dot_names():
    with pytest.raises(SecurityValidationError):
        sanitize_filename("")
    with pytest.raises(SecurityValidationError):
        sanitize_filename("..")
    with pytest.raises(SecurityValidationError):
        sanitize_filename(None)


@pytest.mark.unit
def test_safe_join_rejects_path_traversal(tmp_path):
    (tmp_path / "uploads").mkdir()
    with pytest.raises(SecurityValidationError):
        safe_join_upload_path(tmp_path / "uploads", "../../evil.pdf")
    with pytest.raises(SecurityValidationError):
        safe_join_upload_path(tmp_path / "uploads", "sub/dir.pdf")
    with pytest.raises(SecurityValidationError):
        safe_join_upload_path(tmp_path / "uploads", "C:\\Windows\\evil.pdf")


@pytest.mark.unit
def test_safe_join_accepts_server_generated_name(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    resolved = safe_join_upload_path(uploads, "123e4567-e89b-12d3-a456-426614174000.pdf")
    assert resolved.parent == uploads.resolve()
    assert is_path_within(uploads, resolved)


@pytest.mark.unit
def test_validate_uuid_rejects_injection_payloads():
    with pytest.raises(SecurityValidationError):
        validate_uuid("'; DROP TABLE documents;--")
    with pytest.raises(SecurityValidationError):
        validate_uuid("not-a-uuid")
    assert validate_uuid("123E4567-E89B-12D3-A456-426614174000") == (
        "123e4567-e89b-12d3-a456-426614174000"
    )


@pytest.mark.unit
def test_validate_identifier_rejects_cypher_injection():
    with pytest.raises(SecurityValidationError):
        validate_identifier("x') MATCH (n) DETACH DELETE n //")
    assert validate_identifier("abc123_DEF-456") == "abc123_DEF-456"


@pytest.mark.unit
def test_clamp_int_bounds_untrusted_numbers():
    assert clamp_int(999, 1, 3, 1) == 3
    assert clamp_int(-5, 1, 3, 1) == 1
    assert clamp_int("nonsense", 1, 3, 2) == 2
    assert clamp_int(None, 1, 3, 2) == 2


@pytest.mark.unit
def test_redact_local_paths_removes_windows_and_posix_paths():
    windows = r"Failed to open C:\Users\alice\secret\data.db"
    posix = "Failed to open /home/alice/secret/data.db"
    assert "C:\\Users" not in redact_local_paths(windows)
    assert "/home/alice" not in redact_local_paths(posix)
    assert "[path]" in redact_local_paths(windows)


@pytest.mark.unit
def test_loopback_detection():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.20")
