"""
NeuralForge — HuggingFace authentication helpers.

Single-purpose module that:

  * Discovers an HF token from the standard locations (env vars and the
    `huggingface-cli login` cache file).
  * Validates the token against the Hub's `whoami` endpoint.
  * Tells the inspector / validator whether a model is reachable for the
    current user *before* a download attempt blows up with a 401.

**Security posture.** The actual token bytes never leave this module:

  * `get_token()` returns the raw token but is private to the auth +
    model-source layer; callers in the API surface only see the result of
    `auth_status()` (a redacted summary).
  * `auth_status()` returns `{authenticated, source, scopes, name}` — no
    token material.
  * Errors are wrapped in `_redact()` to scrub anything that looks like
    a bearer credential before propagating.
  * We refuse to log or return the token from any public function.
  * `set_token_for_session()` exists for tests but never persists.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


# Environment variable names HF tooling supports, in priority order.
_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

# Where `huggingface-cli login` writes the token. The directory respects
# `HF_HOME`; the filename has been stable for years.
_TOKEN_CACHE_PATHS = (
    "~/.cache/huggingface/token",
    "~/.huggingface/token",
)


# ---------------------------------------------------------------------------
# Public dataclasses (no token material)
# ---------------------------------------------------------------------------

@dataclass
class AuthStatus:
    """Result of `auth_status()` — safe to serialize / log / return from API.

    *Never* contains the token itself. The `source` field tells you where
    a present token was discovered, useful for "you're using HF_TOKEN from
    your shell" messaging.
    """
    authenticated: bool
    source:        Optional[str] = None    # "env:HF_TOKEN" | "cache:~/.cache/..." | None
    name:          Optional[str] = None    # username from /api/whoami-v2
    org:           Optional[str] = None    # primary org if any
    scopes:        Optional[List[str]] = None
    error:         Optional[str] = None    # populated when authenticated=False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Token discovery (private)
# ---------------------------------------------------------------------------

# Process-local override used by tests. NEVER persisted.
_session_token: Optional[str] = None


def set_token_for_session(token: Optional[str]) -> None:
    """Override the discovered token for the current process only.

    Used by tests and the `neural login` CLI. Does NOT persist anywhere.
    Pass None to clear.
    """
    global _session_token
    _session_token = token


def _discover_token() -> tuple[Optional[str], Optional[str]]:
    """Return `(token, source_label)` or `(None, None)`. Private."""
    if _session_token:
        return _session_token, "session"
    for var in _TOKEN_ENV_VARS:
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip(), f"env:{var}"
    for path in _TOKEN_CACHE_PATHS:
        p = Path(os.path.expanduser(path))
        if p.exists() and p.is_file():
            try:
                token = p.read_text().strip()
                if token:
                    return token, f"cache:{path}"
            except Exception:
                continue
    return None, None


def get_token() -> Optional[str]:
    """**Internal use only.** Return the discovered token or None.

    Callers in the API or UI layer must NOT use this — it returns the raw
    secret. Use `auth_status()` for any user-facing flow.
    """
    return _discover_token()[0]


# ---------------------------------------------------------------------------
# Token validation (talks to the Hub)
# ---------------------------------------------------------------------------

_HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
_HF_TIMEOUT_S = 6.0


def auth_status(timeout_s: float = _HF_TIMEOUT_S) -> AuthStatus:
    """Probe the discovered token against `/api/whoami-v2`.

    Network failures degrade to `authenticated=True` if a token was found
    locally — we don't want to gate offline use on Hub reachability. The
    `error` field carries the diagnostic for the UI to surface.
    """
    token, source = _discover_token()
    if not token:
        return AuthStatus(authenticated=False, source=None,
                          error="No HF token found in HF_TOKEN env var, "
                                "HUGGING_FACE_HUB_TOKEN, or ~/.cache/huggingface/token.")
    try:
        import httpx
    except ImportError:
        return AuthStatus(authenticated=True, source=source,
                          error="httpx is not installed — cannot validate token. "
                                "Token will be used at download time.")
    try:
        with httpx.Client(timeout=timeout_s,
                          headers={"Authorization": f"Bearer {token}",
                                   "User-Agent": "neuralforge/0.3"}) as client:
            r = client.get(_HF_WHOAMI_URL)
            if r.status_code == 401:
                return AuthStatus(authenticated=False, source=source,
                                  error="Token rejected (401). It may be expired or revoked. "
                                        "Re-run `huggingface-cli login` or update HF_TOKEN.")
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        return AuthStatus(authenticated=True, source=source,
                          error=f"Could not reach the Hub to validate the token: {_redact(str(exc))}")
    name = data.get("name") or data.get("fullname")
    org = None
    orgs = data.get("orgs") or []
    if isinstance(orgs, list) and orgs:
        org = orgs[0].get("name") if isinstance(orgs[0], dict) else None
    auth = data.get("auth") or {}
    access_token = auth.get("accessToken") if isinstance(auth, dict) else None
    scopes = None
    if isinstance(access_token, dict):
        role = access_token.get("role")
        scopes = [role] if role else None
    return AuthStatus(authenticated=True, source=source, name=name, org=org, scopes=scopes)


def is_authenticated() -> bool:
    """Cheap sync check — token exists locally, *not* validated against Hub."""
    return bool(get_token())


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Match anything that looks like a bearer token in error / log strings.
# HF tokens look like `hf_<32+ chars>`; we also catch generic Bearer headers.
_TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9_\-\.]{16,}"),
    re.compile(r"(?i)Authorization:\s*Bearer\s+[A-Za-z0-9_\-\.]{16,}"),
    re.compile(r"(?i)token=([A-Za-z0-9_\-\.]{16,})"),
]


def _redact(text: str) -> str:
    """Scrub anything that looks like an HF token from text. Used on every
    error message that might be surfaced in API responses, logs, or the UI."""
    if not text:
        return text
    out = text
    for pat in _TOKEN_PATTERNS:
        out = pat.sub("***REDACTED***", out)
    return out


def redact(text: str) -> str:
    """Public alias of `_redact` so other modules can import it."""
    return _redact(text)
