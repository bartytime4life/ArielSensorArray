# /src/server/api/authz.py
# =============================================================================
# 🔐 SpectraMind V50 — FastAPI AuthZ API (Thin Shim over src.server.authz)
# -----------------------------------------------------------------------------
# Purpose
#   This module provides a *thin API layer* on top of the core authorization
#   utilities defined in `src.server.authz`. It re-exports the FastAPI
#   dependencies (require_roles, require_scopes, etc.) and registers a small
#   administrative router for health, introspection, and live cache reloads.
#
# Design
#   • Single source of truth for auth logic lives in `src.server.authz`.
#   • This module avoids duplication: imports + re-exports dependencies.
#   • Provides minimal endpoints under /api/authz/* for:
#       - Health/introspection
#       - Current user echo
#       - Safe policy view (redacted)
#       - Hot reload (env/policy file changes)
#
# Security
#   • /api/authz/health          → no role needed (optionally IP-guarded).
#   • /api/authz/me              → requires an authenticated (or dev) user.
#   • /api/authz/policy          → requires "admin" role.
#   • /api/authz/reload          → requires "admin" role.
#
# Notes
#   • Keep this module *small* and *stable*. All heavy logic belongs in
#     `src.server.authz`.
#   • Works in OFF mode (dev): returns a synthetic admin user.
#   • HEADER_API_KEY mode: enforces keys/scopes/roles per policy.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from fastapi import APIRouter, Depends, HTTPException, status

# --- Import the core authorization surface from src.server.authz ---------------
from src.server.authz import (  # re-exported below
    User,
    get_current_user,
    get_policy,
    reset_authz_caches,
    require_any_role,
    require_any_scope,
    require_ip_allowlist,
    require_readonly_guard,
    require_rate_limit,
    require_roles,
    require_scopes,
)

# Public router for small administrative endpoints
router = APIRouter(prefix="/api/authz", tags=["authz"])


# -----------------------------------------------------------------------------
# Re-exports (so app modules can import from src.server.api.authz uniformly)
# -----------------------------------------------------------------------------
__all__ = [
    # types
    "User",
    # dependencies
    "get_current_user",
    "require_roles",
    "require_any_role",
    "require_scopes",
    "require_any_scope",
    "require_ip_allowlist",
    "require_readonly_guard",
    "require_rate_limit",
    # utilities
    "router",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _redact_policy_dict(policy: Any) -> Dict[str, Any]:
    """
    Convert Policy to a redacted dict safe for API responses.
    Removes API key material and any sensitive raw fields.
    """
    try:
        users: List[Dict[str, Any]] = []
        for u in policy.users.values():
            # Avoid exposing `api_key` and arbitrary meta that could be sensitive
            users.append(
                {
                    "id": u.id,
                    "name": u.name,
                    "roles": list(u.roles),
                    "scopes": list(u.scopes),
                    # Note: meta intentionally excluded from exposure by default
                }
            )
        roles: Dict[str, Dict[str, Any]] = {}
        for rname, rdef in policy.roles.items():
            roles[rname] = {"scopes": list(rdef.scopes)}

        # Only expose non-sensitive high-level fields
        return {
            "mode": getattr(policy, "_mode", "OFF"),
            "default_role": getattr(policy, "default_role", None),
            "users": users,
            "roles": roles,
            "stats": {
                "num_users": len(users),
                "num_roles": len(roles),
            },
        }
    except Exception as e:
        # Never crash admin endpoints; return minimal info on failure
        return {"error": f"failed to serialize policy: {e.__class__.__name__}"}


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.get(
    "/health",
    summary="AuthZ health & mode",
    description="Returns current authorization mode and basic policy stats.",
)
async def authz_health(
    _ip: None = Depends(require_ip_allowlist()),  # optional IP guard if configured
) -> Dict[str, Any]:
    pol = get_policy()
    return {
        "ok": True,
        "mode": getattr(pol, "_mode", "OFF"),
        "default_role": pol.default_role,
        "users": len(pol.users),
        "roles": len(pol.roles),
    }


@router.get(
    "/me",
    summary="Current user",
    description="Echo the current user resolved by the authorization dependency.",
)
async def authz_me(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "id": user.id,
        "name": user.name,
        "roles": user.roles,
        "scopes": user.scopes,
    }


@router.get(
    "/policy",
    summary="Redacted policy",
    description="Admin-only: view a redacted snapshot of the effective policy "
    "(no API keys or sensitive fields).",
)
async def authz_policy_admin(
    _user: User = Depends(require_roles("admin")),
) -> Dict[str, Any]:
    pol = get_policy()
    return _redact_policy_dict(pol)


@router.post(
    "/reload",
    summary="Reload policy caches",
    description="Admin-only: hot-reload ENV and policy file caches.",
    status_code=status.HTTP_200_OK,
)
async def authz_reload(
    _user: User = Depends(require_roles("admin")),
) -> Dict[str, Any]:
    reset_authz_caches()
    # Touch get_policy() to repopulate
    pol = get_policy()
    return {
        "ok": True,
        "mode": getattr(pol, "_mode", "OFF"),
        "message": "authorization caches reloaded",
    }


# -----------------------------------------------------------------------------
# Example usage notes (kept here for developers; not served as endpoints)
# -----------------------------------------------------------------------------
# In your FastAPI app factory:
#
#   from src.server.api.authz import router as authz_router
#   app.include_router(authz_router)
#
# Protecting endpoints:
#
#   from fastapi import Depends
#   from src.server.api.authz import require_scopes, require_roles, get_current_user
#
#   @app.get("/api/diagnostics/summary")
#   async def summary_endpoint(
#       user = Depends(require_scopes("diagnostics:read"))
#   ):
#       ...
#
# Read-only guard / rate limit / IP allow:
#
#   @app.post("/api/expensive")
#   async def expensive_op(
#       _ip = Depends(require_ip_allowlist()),
#       _ro = Depends(require_readonly_guard()),
#       _rl = Depends(require_rate_limit()),
#       user = Depends(get_current_user),
#   ):
#       ...
#
# Keep the core logic in `src.server.authz` up-to-date; this API shim should remain thin.
