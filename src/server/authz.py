# src/server/authz.py
# =============================================================================
# 🔐 SpectraMind V50 — Minimal, Configurable Authorization Layer (FastAPI)
# -----------------------------------------------------------------------------
# Goals
#   • Keep GUI optional and CLI-first: this authz layer guards server routes,
#     not the CLI itself. It’s intentionally simple, local-first, and file/env
#     configurable so it works air-gapped and inside CI/Kaggle.
#   • Provide a clean FastAPI dependency for role/scope checks without pulling
#     in a heavyweight IAM stack.
#
# Features
#   • Modes:
#       - OFF (default): allow all requests (useful for local dev).
#       - HEADER_API_KEY: check X-API-Key (or Authorization: Bearer …) against
#         an in-memory policy loaded from ENV or a local JSON/YAML file.
#   • Policy schema (users, roles, scopes) with simple RBAC/ABAC helpers.
#   • FastAPI dependencies:
#       - get_current_user()
#       - require_roles(*roles) / require_any_role(*roles)
#       - require_scopes(*scopes) / require_any_scope(*scopes)
#   • Zero network calls; suitable for offline, reproducible setups.
#
# Configuration (environment variables)
#   AUTHZ_MODE                : "OFF" | "HEADER_API_KEY" (default: "OFF")
#   AUTHZ_POLICY_FILE         : Path to JSON/YAML file with policy (optional)
#   AUTHZ_API_KEYS_JSON       : Inline JSON mapping api_key -> user object
#                               e.g. {"abc123":{"id":"dev","roles":["admin"],"scopes":["*"]}}
#   AUTHZ_DEFAULT_ROLE        : Role to assign if user not found (optional)
#   AUTHZ_ACCEPT_BEARER       : "1" to accept Authorization: Bearer <token> as API key
#
# Policy file format (JSON or YAML):
# {
#   "users": [
#     {"id": "dev", "name": "Developer", "api_key": "abc123", "roles": ["admin"], "scopes": ["*"]},
#     {"id": "viewer", "api_key": "viewkey", "roles": ["read"], "scopes": ["diagnostics:read"]}
#   ],
#   "roles": {
#     "admin": {"scopes": ["*"]},
#     "read":  {"scopes": ["diagnostics:read"]}
#   }
# }
#
# Example usage in FastAPI:
#   from fastapi import FastAPI, Depends
#   from src.server.authz import get_current_user, require_roles
#
#   app = FastAPI()
#   @app.get("/api/secret")
#   def secret(user = Depends(require_roles("admin"))):
#       return {"ok": True, "user": user.id}
# =============================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

# Optional YAML support (only if file extension is .yaml/.yml)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class RoleDef:
    name: str
    scopes: List[str] = field(default_factory=list)


@dataclass
class User:
    id: str
    name: Optional[str] = None
    api_key: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_any_role(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)

    def has_scope(self, scope: str, role_index: Mapping[str, RoleDef]) -> bool:
        if "*" in self.scopes:
            return True
        if scope in self.scopes:
            return True
        # scopes inherited from roles
        for r in self.roles:
            rd = role_index.get(r)
            if not rd:
                continue
            if "*" in rd.scopes or scope in rd.scopes:
                return True
        return False

    def has_any_scope(self, scopes: Iterable[str], role_index: Mapping[str, RoleDef]) -> bool:
        return any(self.has_scope(s, role_index) for s in scopes)


class _PolicyModel(BaseModel):
    users: List[Dict[str, Any]] = []
    roles: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Configuration & Policy Loading
# -----------------------------------------------------------------------------

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key, default)
    return v


def _load_policy_from_file(path: Path) -> _PolicyModel:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"AUTHZ_POLICY_FILE not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed but AUTHZ_POLICY_FILE is YAML")
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text or "{}")
    return _PolicyModel(**raw)


def _load_policy_from_env() -> _PolicyModel:
    inline_json = _env("AUTHZ_API_KEYS_JSON", "")
    if inline_json:
        raw = json.loads(inline_json)
        # Normalize into policy model
        users = []
        roles = {}
        for api_key, u in raw.items():
            if not isinstance(u, dict):
                continue
            u = {**u}
            u["api_key"] = api_key
            users.append(u)
        return _PolicyModel(users=users, roles=roles)
    return _PolicyModel()  # empty policy


@dataclass
class Policy:
    users: Dict[str, User]
    roles: Dict[str, RoleDef]
    default_role: Optional[str] = None

    @classmethod
    def from_model(cls, pm: _PolicyModel, default_role: Optional[str]) -> "Policy":
        users: Dict[str, User] = {}
        for u in pm.users:
            users[(u.get("api_key") or "").strip()] = User(
                id=str(u.get("id") or u.get("name") or "unknown"),
                name=u.get("name"),
                api_key=(u.get("api_key") or "").strip() or None,
                roles=list(u.get("roles") or []),
                scopes=list(u.get("scopes") or []),
                meta={k: v for k, v in u.items() if k not in {"id", "name", "api_key", "roles", "scopes"}},
            )
        roles: Dict[str, RoleDef] = {}
        for rname, rdef in pm.roles.items():
            roles[rname] = RoleDef(name=rname, scopes=list(rdef.get("scopes") or []))
        return cls(users=users, roles=roles, default_role=default_role)

    def get_by_api_key(self, api_key: str) -> Optional[User]:
        api_key = (api_key or "").strip()
        if not api_key:
            return None
        user = self.users.get(api_key)
        if user:
            return user
        # If not found, but default_role is set, create ephemeral user.
        if self.default_role:
            return User(id=f"anon:{api_key[:6]}", api_key=api_key, roles=[self.default_role], scopes=[])
        return None


@lru_cache(maxsize=1)
def get_policy() -> Policy:
    mode = (_env("AUTHZ_MODE", "OFF") or "OFF").upper()
    default_role = _env("AUTHZ_DEFAULT_ROLE", None)

    # Build an empty base policy
    base = _PolicyModel()

    # Overlay from env JSON first (highest precedence for quick overrides)
    env_model = _load_policy_from_env()

    # Overlay from file if present
    file_path = _env("AUTHZ_POLICY_FILE", None)
    if file_path:
        try:
            fm = _load_policy_from_file(Path(file_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load AUTHZ_POLICY_FILE: {e}") from e
    else:
        fm = _PolicyModel()

    # Merge: file then env (env wins if duplicate api_key)
    merged_users = { (u.get("api_key") or "").strip(): u for u in (fm.users or []) }
    for u in env_model.users:
        merged_users[(u.get("api_key") or "").strip()] = u
    merged_roles = dict(fm.roles or {})
    # no env roles today (could add AUTHZ_ROLES_JSON if needed)

    merged = _PolicyModel(users=list(merged_users.values()), roles=merged_roles)
    policy = Policy.from_model(merged, default_role=default_role)
    # Cache note: We intentionally *do not* cache by mode; callers rely on mode below.
    policy._mode = mode  # type: ignore[attr-defined]
    return policy


def _mode() -> str:
    return getattr(get_policy(), "_mode", "OFF")  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# FastAPI dependencies
# -----------------------------------------------------------------------------

class _AuthzError(HTTPException):
    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(status_code=status_code, detail=detail)


def _extract_api_key(x_api_key: Optional[str], authorization: Optional[str]) -> Optional[str]:
    """
    Accept either:
      • X-API-Key: <key>
      • Authorization: Bearer <token>       (if AUTHZ_ACCEPT_BEARER=1)
    """
    if x_api_key:
        return x_api_key.strip()
    accept_bearer = (_env("AUTHZ_ACCEPT_BEARER", "1") or "1") in {"1", "true", "TRUE"}
    if accept_bearer and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


async def get_current_user(
    x_api_key: Optional[str] = Header(default=None, convert_underscores=False),
    authorization: Optional[str] = Header(default=None),
) -> User:
    """
    Basic authentication dependency. Behavior per AUTHZ_MODE:

      OFF:
        • Returns a synthetic admin user "dev" with wildcard scope, for local dev.
      HEADER_API_KEY:
        • Requires X-API-Key (or Authorization: Bearer <token> if allowed).
        • Looks up user in policy (AUTHZ_POLICY_FILE / AUTHZ_API_KEYS_JSON).
        • Returns 401 if missing/unknown.

    Raises:
      401 Unauthorized if key missing/invalid (in HEADER_API_KEY mode).
    """
    mode = _mode().upper()
    if mode == "OFF":
        return User(id="dev", name="Developer", roles=["admin"], scopes=["*"])

    if mode == "HEADER_API_KEY":
        api_key = _extract_api_key(x_api_key, authorization)
        if not api_key:
            raise _AuthzError(status.HTTP_401_UNAUTHORIZED, "Missing API key")
        user = get_policy().get_by_api_key(api_key)
        if not user:
            raise _AuthzError(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
        return user

    raise _AuthzError(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Unsupported AUTHZ_MODE: {mode}")


def require_roles(*roles: str):
    """
    Dependency factory to require ALL specified roles.
    Example:
      @app.get("/admin")
      def admin(_=Depends(require_roles("admin"))):
          return {"ok": True}
    """
    roles_set = set(roles)

    async def _dep(user: User = Depends(get_current_user)) -> User:
        missing = [r for r in roles_set if r not in user.roles]
        if missing:
            raise _AuthzError(
                status.HTTP_403_FORBIDDEN,
                {"msg": "Missing required role(s)", "missing": missing, "user_roles": user.roles},
            )
        return user

    return _dep


def require_any_role(*roles: str):
    """
    Dependency factory to require ANY of the specified roles.
    """
    roles_set = set(roles)

    async def _dep(user: User = Depends(get_current_user)) -> User:
        if not any(r in user.roles for r in roles_set):
            raise _AuthzError(
                status.HTTP_403_FORBIDDEN,
                {"msg": "Requires any of roles", "roles": list(roles_set), "user_roles": user.roles},
            )
        return user

    return _dep


def require_scopes(*scopes: str):
    """
    Dependency factory to require ALL specified scopes (including role-inherited).
    Scopes support wildcard '*' for superuser in either user.scopes or role scopes.
    """
    scopes_list: List[str] = list(scopes)

    async def _dep(user: User = Depends(get_current_user)) -> User:
        role_index = get_policy().roles
        missing = [s for s in scopes_list if not user.has_scope(s, role_index)]
        if missing:
            raise _AuthzError(
                status.HTTP_403_FORBIDDEN,
                {"msg": "Missing required scope(s)", "missing": missing},
            )
        return user

    return _dep


def require_any_scope(*scopes: str):
    """
    Dependency factory to require ANY of the specified scopes.
    """
    scopes_list: List[str] = list(scopes)

    async def _dep(user: User = Depends(get_current_user)) -> User:
        role_index = get_policy().roles
        if not user.has_any_scope(scopes_list, role_index):
            raise _AuthzError(
                status.HTTP_403_FORBIDDEN,
                {"msg": "Requires any of scopes", "scopes": scopes_list},
            )
        return user

    return _dep


# -----------------------------------------------------------------------------
# Utilities for programmatic checks (non-FastAPI contexts)
# -----------------------------------------------------------------------------

def check_roles(user: User, *roles: str) -> None:
    missing = [r for r in roles if r not in user.roles]
    if missing:
        raise _AuthzError(status.HTTP_403_FORBIDDEN, {"msg": "Missing roles", "missing": missing})


def check_scopes(user: User, *scopes: str) -> None:
    role_index = get_policy().roles
    missing = [s for s in scopes if not user.has_scope(s, role_index)]
    if missing:
        raise _AuthzError(status.HTTP_403_FORBIDDEN, {"msg": "Missing scopes", "missing": missing})


# -----------------------------------------------------------------------------
# Inline smoke test helper (optional)
# -----------------------------------------------------------------------------

def _example_policy() -> str:
    return json.dumps(
        {
            "users": [
                {"id": "dev", "name": "Developer", "api_key": "abc123", "roles": ["admin"], "scopes": ["*"]},
                {"id": "viewer", "api_key": "viewkey", "roles": ["read"], "scopes": ["diagnostics:read"]},
            ],
            "roles": {
                "admin": {"scopes": ["*"]},
                "read": {"scopes": ["diagnostics:read"]},
            },
        },
        indent=2,
    )


if __name__ == "__main__":
    # Minimal manual test:
    #   AUTHZ_MODE=HEADER_API_KEY AUTHZ_API_KEYS_JSON='{"abc123":{"id":"dev","roles":["admin"],"scopes":["*"]}}' \
    #   python -m src.server.authz
    os.environ.setdefault("AUTHZ_MODE", "HEADER_API_KEY")
    os.environ.setdefault(
        "AUTHZ_API_KEYS_JSON",
        '{"abc123":{"id":"dev","name":"Developer","roles":["admin"],"scopes":["*"]}}',
    )
    p = get_policy()
    u = p.get_by_api_key("abc123")
    print("Loaded policy users:", list(p.users.keys()))
    print("User:", u.id if u else None, "roles:", u.roles if u else None, "scopes:", u.scopes if u else None)
