"""Same-origin gate for mutating Motion Context HTTP routes.

The Comfy Registry flags unauthenticated POST/DELETE that write or delete
files as CSRF-forgeable (policy-v0.2 UNAUTHENTICATED_SIDE_EFFECT). ComfyUI's
own loopback Origin/Host middleware is not part of this pack, so the
clear-latents handler must check the request itself.

A same-origin UI fetch sends Origin matching Host. A cross-site form POST
sends a different Origin, or Origin: null from a sandboxed iframe.
"""

from functools import wraps
from urllib.parse import urlparse, urlsplit

from aiohttp import web

_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _origin_parts(value):
    raw = (value or "").strip()
    if not raw or raw.lower() == "null":
        return "", None
    parsed = urlparse(raw)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return "", None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return hostname, port


def _host_parts(host_header):
    parsed = urlsplit("//" + (host_header or "").strip().lower())
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    if port is None:
        port = 80
    return hostname, port


def reject_csrf(request):
    """Return a 403 response if this mutating request is not same-origin."""
    if request.method not in _MUTATING:
        return None

    site = (request.headers.get("Sec-Fetch-Site") or "").strip().lower()
    if site == "cross-site":
        return web.json_response(
            {"error": "Cross-site request blocked."}, status=403)

    host = (request.headers.get("Host") or "").strip()
    hh, hp = _host_parts(host)
    if not hh:
        return web.json_response({"error": "Missing Host."}, status=403)

    origin_raw = (request.headers.get("Origin") or "").strip()
    oh, op = _origin_parts(origin_raw)
    if not oh:
        oh, op = _origin_parts(request.headers.get("Referer") or "")
        if not oh:
            return web.json_response(
                {"error": "Missing Origin. Same-origin UI requests send it."},
                status=403,
            )

    if oh != hh or op != hp:
        return web.json_response(
            {"error": "Origin does not match Host."}, status=403)
    return None


def require_same_origin(handler):
    """Decorator: reject CSRF-forgeable unauthenticated POST/DELETE."""
    @wraps(handler)
    async def wrapped(request):
        blocked = reject_csrf(request)
        if blocked is not None:
            return blocked
        return await handler(request)
    return wrapped
