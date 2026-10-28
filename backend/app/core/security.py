"""Security helpers shared across API and ingestion layers.

These helpers are deliberately small and side-effect free so they can be
covered by fast unit tests. They do not depend on FastAPI or any database.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Optional

# Control characters that can break logs / filenames. Newlines and tabs are
# intentionally excluded because they are meaningful in document text.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Stored files are always server generated. Keep this strict so a filename can
# never contain a path separator, drive letter, or traversal segment.
_SAFE_STORED_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Absolute Windows (C:\...) and POSIX (/home/...) paths. Used to keep local
# filesystem details out of HTTP error details.
_ABSOLUTE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s,;\"']*|(?:^|\s)/(?:[^\s/]+/)+[^\s,;\"']*)"
)


class SecurityValidationError(ValueError):
    """Raised when untrusted input fails a security check."""


def sanitize_filename(filename: Optional[str], max_length: int = 255) -> str:
    """Return a display-safe basename for an uploaded file.

    The value is only used for metadata and UI display. Stored files always use
    a server-generated UUID name.
    """
    if not filename:
        raise SecurityValidationError("Filename is required")

    # Treat both separators as path separators so a Windows upload cannot
    # smuggle a traversal segment through a POSIX-style basename call.
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    name = _CONTROL_CHARS.sub("", name).strip()
    if not name or name in {".", ".."}:
        raise SecurityValidationError("Invalid filename")
    if len(name) > max_length:
        stem, dot, extension = name.rpartition(".")
        if dot and len(extension) <= 16:
            keep = max_length - len(extension) - 1
            name = f"{stem[:keep]}.{extension}"
        else:
            name = name[:max_length]
    return name


def validate_uuid(value: str, label: str = "document ID") -> str:
    """Validate and normalize a UUID used in database queries.

    Returning the canonical string form means malformed IDs are rejected before
    they ever reach PostgreSQL, where a failed cast could otherwise surface as a
    500 and leak database error details.
    """
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise SecurityValidationError(f"Invalid {label}")
    return str(parsed)


def validate_identifier(value: str, label: str = "identifier") -> str:
    """Validate a short opaque identifier (entity/chunk IDs)."""
    if not value or not _SAFE_IDENTIFIER.fullmatch(str(value)):
        raise SecurityValidationError(f"Invalid {label}")
    return str(value)


def clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    """Clamp an integer to a safe range, falling back on invalid input."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def safe_join_upload_path(upload_dir: str | Path, stored_filename: str) -> Path:
    """Resolve a stored filename inside the upload directory.

    Raises SecurityValidationError for path traversal, absolute paths, or
    names that are not strict server-generated filenames.
    """
    if not _SAFE_STORED_FILENAME.fullmatch(stored_filename or ""):
        raise SecurityValidationError("Invalid stored filename")
    if ".." in stored_filename or "/" in stored_filename or "\\" in stored_filename:
        raise SecurityValidationError("Invalid stored filename")

    base = Path(upload_dir).resolve()
    candidate = (base / stored_filename).resolve()
    if candidate.parent != base:
        raise SecurityValidationError("Invalid file path")
    return candidate


def is_path_within(base_dir: str | Path, target: str | Path) -> bool:
    """Return True when target resolves inside base_dir."""
    try:
        base = Path(base_dir).resolve()
        candidate = Path(target).resolve()
        candidate.relative_to(base)
        return True
    except (ValueError, OSError):
        return False


def remove_file_within(base_dir: str | Path, target: str | Path) -> bool:
    """Delete a file only when it is inside base_dir and is not a symlink."""
    try:
        base = Path(base_dir).resolve()
        candidate = Path(target)
        if candidate.is_symlink():
            return False
        resolved = candidate.resolve()
        resolved.relative_to(base)
        if resolved.is_file():
            resolved.unlink()
            return True
    except (ValueError, OSError):
        return False
    return False


def redact_local_paths(message: str, max_length: int = 300) -> str:
    """Remove absolute paths from an error message before returning it."""
    redacted = _ABSOLUTE_PATH.sub("[path]", str(message))
    redacted = _CONTROL_CHARS.sub("", redacted).strip()
    if len(redacted) > max_length:
        redacted = redacted[:max_length].rstrip() + "..."
    return redacted or "Request could not be processed"


def is_loopback_host(host: str) -> bool:
    """Return True for addresses that are not reachable from the LAN."""
    normalized = (host or "").strip().lower().strip("[]")
    return normalized in {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}
