# src/server/authz.py

# =============================================================================

# 🔐 SpectraMind V50 — Minimal, Configurable Authorization Layer (FastAPI)

# -----------------------------------------------------------------------------

# Goals

# • Keep GUI optional and CLI-first: this authz layer guards server routes,

# not the CLI itself. It’s intentionally simple, local-first, and file/env

# configurable so it works air-gapped and inside CI/Kaggle.

# • Provide a clean FastAPI dependency for role/scope checks without pulling

# in a heavyweight IAM stack.

#

# Upgrades in this version

# • Hot-reloadable policy cache (file mtime + env fingerprint) with reset hook.

# • Wildcard + glob scope matching (e.g., "diagnostics:\*", "submit:?ead").

# • Optional IP allow/deny with CIDR support (AUTHZ\_IP\_ALLOW / AUTHZ\_IP\_DENY).

# • Optional read-only guard (block mutating methods unless allowed).

# • Simple in-memory, thread-safe rate limiting (token bucket per API key/IP).

# • JSONL audit log for allow/deny decisions (AUTHZ\_AUDIT\_LOG path).

# • Dependency helpers compose cleanly: get\_current\_user, require\_\*(),

# require\_rate\_limit(), require\_readonly\_guard().

#

# Features

# • Modes:

# - OFF (default): allow all requests (useful for local dev).

# - HEADER\_API\_KEY: check X-API-Key (or Authorization: Bearer …) against

# an in-memory policy loaded from ENV or a local JSON/YAML file.

# • Policy schema (users, roles, scopes) with simple RBAC/ABAC helpers.

# • FastAPI dependencies:

# - get\_current\_user()

# - require\_roles(\*roles) / require\_any\_role(\*roles)

# - require\_scopes(\*scopes) / require\_any\_scope(\*scopes)

# - require\_ip\_allowlist()            # optional IP guard

# - require\_readonly\_guard()          # optional read-only protection

# - require\_rate\_limit()              # optional rate limit guard

# • Zero network calls; suitable for offline, reproducible setups.

#

# Configuration (environment variables)

# AUTHZ\_MODE                : "OFF" | "HEADER\_API\_KEY" (default: "OFF")

# AUTHZ\_POLICY\_FILE         : Path to JSON/YAML file with policy (optional)

# AUTHZ\_API\_KEYS\_JSON       : Inline JSON mapping api\_key -> user object

# e.g. {"abc123":{"id":"dev","roles":\["admin"],"scopes":\["\*"]}}

# AUTHZ\_DEFAULT\_ROLE        : Role to assign if user not found (optional)

# AUTHZ\_ACCEPT\_BEARER       : "1" to accept Authorization: Bearer <token> as API key (default "1")

# AUTHZ\_IP\_ALLOW            : Comma list of IPs/CIDRs allowed (optional)

# AUTHZ\_IP\_DENY             : Comma list of IPs/CIDRs denied  (optional)

# AUTHZ\_READONLY            : "1" to block mutating HTTP methods unless user has scope "write:\*" or role grants it

# AUTHZ\_AUDIT\_LOG           : Path to JSONL audit log file (optional). If set, decisions are appended here.

# AUTHZ\_RATE\_LIMIT          : Token bucket "N/period" (e.g., "60/min", "300/5min", "1000/hour"). Optional.

# AUTHZ\_RATE\_BURST          : Optional burst capacity (int). Defaults to N from AUTHZ\_RATE\_LIMIT.

#

# Policy file format (JSON or YAML):

# {

# "users": \[

# {"id": "dev", "name": "Developer", "api\_key": "abc123", "roles": \["admin"], "scopes": \["\*"]},

# {"id": "viewer", "api\_key": "viewkey", "roles": \["read"], "scopes": \["diagnostics\:read"]}

# ],

# "roles": {

# "admin": {"scopes": \["\*"]},

# "read":  {"scopes": \["diagnostics\:read"]}

# }

# }

#

# Example usage in FastAPI:

# from fastapi import FastAPI, Depends

# from src.server.authz import (

# get\_current\_user, require\_roles, require\_scopes,

# require\_ip\_allowlist, require\_readonly\_guard, require\_rate\_limit

# )

#

# app = FastAPI()

# @app.get("/api/secret")

# def secret(user = Depends(require\_roles("admin"))):

# return {"ok": True, "user": user.id}

#

# @app.post("/api/launch")

# def launch(

# \_ip   = Depends(require\_ip\_allowlist()),

# \_ro   = Depends(require\_readonly\_guard()),

# \_rl   = Depends(require\_rate\_limit()),

# user  = Depends(require\_scopes("pipeline\:run", "write:\*")),

# ):

# return {"ok": True, "user": user.id}

# =============================================================================

from **future** import annotations

import ipaddress
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru\_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from fastapi import Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

# Optional YAML support (only if file extension is .yaml/.yml)

try:
import yaml  # type: ignore
except Exception:  # pragma: no cover
yaml = None

**all** = \[
"User",
"RoleDef",
"get\_current\_user",
"require\_roles",
"require\_any\_role",
"require\_scopes",
"require\_any\_scope",
"require\_ip\_allowlist",
"require\_readonly\_guard",
"require\_rate\_limit",
"check\_roles",
"check\_scopes",
"reset\_authz\_caches",
]

# -----------------------------------------------------------------------------

# Data models

# -----------------------------------------------------------------------------

@dataclass
class RoleDef:
name: str
scopes: List\[str] = field(default\_factory=list)

@dataclass
class User:
id: str
name: Optional\[str] = None
api\_key: Optional\[str] = None
roles: List\[str] = field(default\_factory=list)
scopes: List\[str] = field(default\_factory=list)
meta: Dict\[str, Any] = field(default\_factory=dict)

```
def has_role(self, role: str) -> bool:
    return role in self.roles

def has_any_role(self, *roles: str) -> bool:
    return any(r in self.roles for r in roles)

def _match_scope_token(self, pattern: str, candidate: str) -> bool:
    """
    Glob-like matching for a single scope string.
    Supports:
      * '*' matches everything
      * '?' single character
      * character classes and alternatives via fnmatch-like patterns are emulated with regex
    """
    if pattern == "*" or pattern == candidate:
        return True
    # Convert simple glob to regex
    regex = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
    return re.match(regex, candidate) is not None

def has_scope(self, scope: str, role_index: Mapping[str, RoleDef]) -> bool:
    # Direct user scopes (with wildcard support)
    for s in self.scopes:
        if self._match_scope_token(s, scope) or s == "*":
            return True
    # scopes inherited from roles (with wildcard support)
    for r in self.roles:
        rd = role_index.get(r)
        if not rd:
            continue
        for s in rd.scopes:
            if s == "*" or self._match_scope_token(s, scope):
                return True
    return False

def has_any_scope(self, scopes: Iterable[str], role_index: Mapping[str, RoleDef]) -> bool:
    return any(self.has_scope(s, role_index) for s in scopes)
```

class \_PolicyModel(BaseModel):
users: List\[Dict\[str, Any]] = \[]
roles: Dict\[str, Dict\[str, Any]] = {}

# -----------------------------------------------------------------------------

# Configuration helpers

# -----------------------------------------------------------------------------

\_ENV\_LOCK = threading.Lock()

def \_env(key: str, default: Optional\[str] = None) -> Optional\[str]:
return os.getenv(key, default)

def \_bool\_env(key: str, default\_true: bool = True) -> bool:
raw = (\_env(key, None) or "").strip().lower()
if not raw:
return default\_true
return raw in {"1", "true", "yes", "on"}

def \_rate\_env() -> Optional\[Tuple\[int, float]]:
"""
Parse AUTHZ\_RATE\_LIMIT = "N/period", where period ∈ {sec, second(s), min, minute(s), hour(s)}
Returns (tokens\_per\_period, period\_seconds) or None.
"""
value = (\_env("AUTHZ\_RATE\_LIMIT") or "").strip()
if not value:
return None
try:
count\_str, period\_str = value.split("/", 1)
n = int(count\_str.strip())
p = period\_str.strip().lower()
if p in {"sec", "second", "seconds"}:
secs = 1.0
elif p in {"min", "mins", "minute", "minutes"}:
secs = 60.0
elif p in {"hour", "hours", "hr", "hrs"}:
secs = 3600.0
elif p.endswith("min"):
\# e.g., "5min"
m = int(p.replace("min", "").strip())
secs = 60.0 \* m
elif p.endswith("hour") or p.endswith("hr"):
h = int(re.sub(r"(hour|hr)s?\$", "", p).strip())
secs = 3600.0 \* h
else:
\# fallback: numeric seconds
secs = float(p)
return (n, secs)
except Exception:
return None

# -----------------------------------------------------------------------------

# Policy Loading (with hot-reload)

# -----------------------------------------------------------------------------

@dataclass
class Policy:
users: Dict\[str, User]
roles: Dict\[str, RoleDef]
default\_role: Optional\[str] = None
\_mode: str = "OFF"
\_file\_mtime: Optional\[float] = None
\_env\_fingerprint: str = ""

```
@classmethod
def from_model(
    cls,
    pm: _PolicyModel,
    default_role: Optional[str],
    mode: str,
    file_mtime: Optional[float],
    env_fp: str,
) -> "Policy":
    users: Dict[str, User] = {}
    for u in pm.users:
        api_key = (u.get("api_key") or "").strip()
        if api_key:
            users[api_key] = User(
                id=str(u.get("id") or u.get("name") or "unknown"),
                name=u.get("name"),
                api_key=api_key,
                roles=list(u.get("roles") or []),
                scopes=list(u.get("scopes") or []),
                meta={k: v for k, v in u.items() if k not in {"id", "name", "api_key", "roles", "scopes"}},
            )
    roles: Dict[str, RoleDef] = {}
    for rname, rdef in pm.roles.items():
        roles[rname] = RoleDef(name=rname, scopes=list(rdef.get("scopes") or []))
    return cls(
        users=users,
        roles=roles,
        default_role=default_role,
        _mode=mode,
        _file_mtime=file_mtime,
        _env_fingerprint=env_fp,
    )

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
```

def \_load\_policy\_from\_file(path: Path) -> Tuple\[\_PolicyModel, Optional\[float]]:
if not path.exists() or not path.is\_file():
raise FileNotFoundError(f"AUTHZ\_POLICY\_FILE not found: {path}")
text = path.read\_text(encoding="utf-8")
mtime = path.stat().st\_mtime
if path.suffix.lower() in {".yaml", ".yml"}:
if yaml is None:
raise RuntimeError("PyYAML not installed but AUTHZ\_POLICY\_FILE is YAML")
raw = yaml.safe\_load(text) or {}
else:
raw = json.loads(text or "{}")
return \_PolicyModel(\*\*raw), mtime

def \_load\_policy\_from\_env() -> \_PolicyModel:
inline\_json = \_env("AUTHZ\_API\_KEYS\_JSON", "")
if inline\_json:
raw = json.loads(inline\_json)
\# Normalize into policy model
users = \[]
roles = {}
for api\_key, u in raw\.items():
if not isinstance(u, dict):
continue
u = {\*\*u}
u\["api\_key"] = api\_key
users.append(u)
return \_PolicyModel(users=users, roles=roles)
return \_PolicyModel()  # empty policy

def \_env\_fingerprint() -> str:
"""
Build a small fingerprint of relevant env vars so we can detect changes and hot-reload.
"""
keys = \[
"AUTHZ\_MODE",
"AUTHZ\_DEFAULT\_ROLE",
"AUTHZ\_POLICY\_FILE",
"AUTHZ\_API\_KEYS\_JSON",
"AUTHZ\_ACCEPT\_BEARER",
"AUTHZ\_IP\_ALLOW",
"AUTHZ\_IP\_DENY",
"AUTHZ\_READONLY",
"AUTHZ\_AUDIT\_LOG",
"AUTHZ\_RATE\_LIMIT",
"AUTHZ\_RATE\_BURST",
]
return "|".join(f"{k}={\_env(k,'')}" for k in keys)

@lru\_cache(maxsize=1)
def \_policy\_cached() -> Policy:
mode = (\_env("AUTHZ\_MODE", "OFF") or "OFF").upper()
default\_role = \_env("AUTHZ\_DEFAULT\_ROLE", None)

```
# Start with empty model
file_model, file_mtime = _PolicyModel(), None

# Load from file if present
file_path = _env("AUTHZ_POLICY_FILE", None)
if file_path:
    fm, mtime = _load_policy_from_file(Path(file_path))
    file_model, file_mtime = fm, mtime

# Overlay from env JSON (wins if duplicate api_key)
env_model = _load_policy_from_env()
merged_users = {(u.get("api_key") or "").strip(): u for u in (file_model.users or [])}
for u in env_model.users or []:
    merged_users[(u.get("api_key") or "").strip()] = u
merged_roles = dict(file_model.roles or {})
merged = _PolicyModel(users=list(merged_users.values()), roles=merged_roles)

policy = Policy.from_model(
    merged,
    default_role=default_role,
    mode=mode,
    file_mtime=file_mtime,
    env_fp=_env_fingerprint(),
)
return policy
```

def get\_policy() -> Policy:
"""
Return policy, hot-reloading if either env fingerprint or policy file mtime changed.
"""
pol = \_policy\_cached()
file\_path = \_env("AUTHZ\_POLICY\_FILE", None)
need\_reset = False

```
# Check env fingerprint
if pol._env_fingerprint != _env_fingerprint():
    need_reset = True

# Check file mtime
if file_path:
    p = Path(file_path)
    if p.exists():
        current_mtime = p.stat().st_mtime
        if pol._file_mtime != current_mtime:
            need_reset = True

if need_reset:
    reset_authz_caches()
    pol = _policy_cached()

return pol
```

def reset\_authz\_caches() -> None:
"""Public hook for tests/admin to clear authorization caches."""
\_policy\_cached.cache\_clear()  # type: ignore\[attr-defined]
\_iplist\_cache.cache\_clear()   # type: ignore\[attr-defined]

def \_mode() -> str:
return get\_policy().\_mode

# -----------------------------------------------------------------------------

# IP allow/deny handling

# -----------------------------------------------------------------------------

@lru\_cache(maxsize=1)
def \_iplist\_cache() -> Tuple\[List\[ipaddress.\_BaseNetwork], List\[ipaddress.\_BaseNetwork]]:
allow\_raw = (\_env("AUTHZ\_IP\_ALLOW") or "").strip()
deny\_raw = (\_env("AUTHZ\_IP\_DENY") or "").strip()

```
def _parse_list(raw: str) -> List[ipaddress._BaseNetwork]:
    nets: List[ipaddress._BaseNetwork] = []
    if not raw:
        return nets
    for item in [s.strip() for s in raw.split(",") if s.strip()]:
        try:
            # Accept single IP as /32 (IPv4) or /128 (IPv6)
            if "/" not in item:
                ip_obj = ipaddress.ip_address(item)
                net = ipaddress.ip_network(f"{item}/32" if ip_obj.version == 4 else f"{item}/128", strict=False)
            else:
                net = ipaddress.ip_network(item, strict=False)
            nets.append(net)
        except Exception:
            continue
    return nets

return _parse_list(allow_raw), _parse_list(deny_raw)
```

def \_client\_ip(request: Request) -> Optional\[ipaddress.\_BaseAddress]:
"""
Extract client IP in a conservative, proxy-agnostic way.
We prefer request.client.host. If behind a trusted proxy, consider adding an
API layer that resolves X-Forwarded-For before FastAPI, not here.
"""
host = request.client.host if request and request.client else None
if not host:
return None
try:
return ipaddress.ip\_address(host)
except Exception:
return None

def \_ip\_allowed(ip: Optional\[ipaddress.\_BaseAddress]) -> bool:
allow, deny = \_iplist\_cache()
if ip is None:
\# If no IP known, default allow unless deny list exists (conservative).
return not bool(deny)
\# Deny has priority
for net in deny:
if ip in net:
return False
\# If allow list specified, require membership
if allow:
return any(ip in net for net in allow)
return True

def require\_ip\_allowlist():
"""
Optional IP allow/deny dependency. If no lists are configured, it allows.
"""
async def \_dep(request: Request) -> None:
if not (\_env("AUTHZ\_IP\_ALLOW") or \_env("AUTHZ\_IP\_DENY")):
return
ip = \_client\_ip(request)
if not \_ip\_allowed(ip):
\_audit\_log(
decision="deny",
reason="ip\_block",
mode=\_mode(),
user=None,
request=request,
)
raise HTTPException(status\_code=status.HTTP\_403\_FORBIDDEN, detail="IP not allowed")
return \_dep

# -----------------------------------------------------------------------------

# Read-only guard (mutating methods)

# -----------------------------------------------------------------------------

\_MUTATING\_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

def require\_readonly\_guard():
"""
If AUTHZ\_READONLY=1, block mutating HTTP methods unless the user has either:
• scope "write:*" (glob allowed), or
• a role that grants a write:* scope
"""
async def \_dep(request: Request, user: User = Depends(get\_current\_user)) -> None:
if not \_bool\_env("AUTHZ\_READONLY", default\_true=False):
return
if request.method not in \_MUTATING\_METHODS:
return
\# Require a generic write scope for mutations; customize as needed.
role\_index = get\_policy().roles
if user.has\_scope("write:*", role\_index) or user.has\_scope("admin:*", role\_index) or user.has\_scope("\*", role\_index):
return
\_audit\_log(
decision="deny",
reason="readonly\_block",
mode=\_mode(),
user=user,
request=request,
extra={"method": request.method},
)
raise HTTPException(status\_code=status.HTTP\_403\_FORBIDDEN, detail="Read-only mode")
return \_dep

# -----------------------------------------------------------------------------

# Rate limiting (token bucket per principal)

# -----------------------------------------------------------------------------

class \_TokenBucket:
def **init**(self, rate\_n: int, per\_seconds: float, burst: Optional\[int] = None) -> None:
self.capacity = burst if burst is not None else rate\_n
self.tokens = float(self.capacity)
self.rate\_per\_sec = float(rate\_n) / float(per\_seconds)
self.updated = time.monotonic()
self.lock = threading.Lock()

```
def allow(self) -> bool:
    with self.lock:
        now = time.monotonic()
        elapsed = now - self.updated
        self.updated = now
        # Refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False
```

\_RATE\_LIMITERS: Dict\[str, \_TokenBucket] = {}
\_RATE\_LOCK = threading.Lock()

def \_rate\_key(request: Request, user: Optional\[User]) -> str:
\# Prefer API key; fallback to IP; then path as minor partition
api\_key = user.api\_key if user and user.api\_key else None
ip = None
cip = \_client\_ip(request)
if cip is not None:
ip = str(cip)
return f"{api\_key or ip or 'anon'}|{request.url.path}"

def require\_rate\_limit():
"""
Optional rate limiter. If AUTHZ\_RATE\_LIMIT is unset, this is a no-op.
The limiter keys by API key (or IP) + path to avoid cross-route starvation.
"""
conf = \_rate\_env()
burst = None
raw\_burst = \_env("AUTHZ\_RATE\_BURST")
if raw\_burst:
try:
burst = int(raw\_burst)
except Exception:
burst = None

```
async def _dep(request: Request, user: Optional[User] = Depends(get_current_user)) -> None:
    if conf is None:
        return
    rate_n, per_seconds = conf
    key = _rate_key(request, user)
    with _RATE_LOCK:
        bucket = _RATE_LIMITERS.get(key)
        if bucket is None:
            bucket = _TokenBucket(rate_n, per_seconds, burst=burst)
            _RATE_LIMITERS[key] = bucket
    if not bucket.allow():
        _audit_log(
            decision="deny",
            reason="rate_limit",
            mode=_mode(),
            user=user,
            request=request,
            extra={"rate": f"{rate_n}/{int(per_seconds)}s", "burst": bucket.capacity},
        )
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
return _dep
```

# -----------------------------------------------------------------------------

# Audit logging

# -----------------------------------------------------------------------------

def \_audit\_log(
\*,
decision: str,              # "allow" | "deny"
reason: str,                # e.g., "ok", "missing\_key", "invalid\_key", "scope", "role"
mode: str,
user: Optional\[User],
request: Optional\[Request],
extra: Optional\[Dict\[str, Any]] = None,
) -> None:
"""
Append a single JSON object to AUTHZ\_AUDIT\_LOG if configured.
This is best-effort and should never raise.
"""
path = \_env("AUTHZ\_AUDIT\_LOG")
if not path:
return
try:
rec = {
"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
"decision": decision,
"reason": reason,
"mode": mode,
"user": {
"id": user.id if user else None,
"roles": user.roles if user else None,
"scopes": user.scopes if user else None,
},
"req": {
"method": getattr(request, "method", None) if request else None,
"path": str(getattr(getattr(request, "url", None), "path", None)) if request else None,
"client\_ip": str(\_client\_ip(request)) if request else None,
},
}
if extra:
rec\["extra"] = extra
with open(path, "a", encoding="utf-8") as f:
f.write(json.dumps(rec, ensure\_ascii=False) + "\n")
except Exception:
\# Silent by design
pass

# -----------------------------------------------------------------------------

# FastAPI dependencies

# -----------------------------------------------------------------------------

class \_AuthzError(HTTPException):
def **init**(self, status\_code: int, detail: Any) -> None:
super().**init**(status\_code=status\_code, detail=detail)

def \_extract\_api\_key(x\_api\_key: Optional\[str], authorization: Optional\[str]) -> Optional\[str]:
"""
Accept either:
• X-API-Key: <key>
• Authorization: Bearer <token>       (if AUTHZ\_ACCEPT\_BEARER=1)
"""
if x\_api\_key:
return x\_api\_key.strip()
accept\_bearer = \_bool\_env("AUTHZ\_ACCEPT\_BEARER", default\_true=True)
if accept\_bearer and authorization:
parts = authorization.split()
if len(parts) == 2 and parts\[0].lower() == "bearer":
return parts\[1].strip()
return None

async def get\_current\_user(
request: Request,
x\_api\_key: Optional\[str] = Header(default=None, convert\_underscores=False),
authorization: Optional\[str] = Header(default=None),
) -> User:
"""
Basic authentication dependency. Behavior per AUTHZ\_MODE:

```
  OFF:
    • Returns a synthetic admin user "dev" with wildcard scope, for local dev.

  HEADER_API_KEY:
    • Requires X-API-Key (or Authorization: Bearer <token> if allowed).
    • Looks up user in policy (AUTHZ_POLICY_FILE / AUTHZ_API_KEYS_JSON).
    • Returns 401 if missing/unknown.

Raises:
  401 Unauthorized if key missing/invalid (in HEADER_API_KEY mode).
"""
policy = get_policy()
mode = policy._mode.upper()

if mode == "OFF":
    user = User(id="dev", name="Developer", roles=["admin"], scopes=["*"])
    _audit_log(decision="allow", reason="mode_off", mode=mode, user=user, request=request)
    return user

if mode == "HEADER_API_KEY":
    api_key = _extract_api_key(x_api_key, authorization)
    if not api_key:
        _audit_log(decision="deny", reason="missing_key", mode=mode, user=None, request=request)
        raise _AuthzError(status.HTTP_401_UNAUTHORIZED, "Missing API key")
    user = policy.get_by_api_key(api_key)
    if not user:
        _audit_log(decision="deny", reason="invalid_key", mode=mode, user=None, request=request)
        raise _AuthzError(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
    _audit_log(decision="allow", reason="ok", mode=mode, user=user, request=request)
    return user

_audit_log(decision="deny", reason=f"unsupported_mode:{mode}", mode=mode, user=None, request=request)
raise _AuthzError(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Unsupported AUTHZ_MODE: {mode}")
```

def require\_roles(\*roles: str):
"""
Dependency factory to require ALL specified roles.
Example:
@app.get("/admin")
def admin(\_=Depends(require\_roles("admin"))):
return {"ok": True}
"""
roles\_set = set(roles)

```
async def _dep(request: Request, user: User = Depends(get_current_user)) -> User:
    missing = [r for r in roles_set if r not in user.roles]
    if missing:
        _audit_log(
            decision="deny",
            reason="role",
            mode=_mode(),
            user=user,
            request=request,
            extra={"missing": missing, "user_roles": user.roles},
        )
        raise _AuthzError(
            status.HTTP_403_FORBIDDEN,
            {"msg": "Missing required role(s)", "missing": missing, "user_roles": user.roles},
        )
    _audit_log(decision="allow", reason="role", mode=_mode(), user=user, request=request, extra={"ok": True})
    return user

return _dep
```

def require\_any\_role(\*roles: str):
"""
Dependency factory to require ANY of the specified roles.
"""
roles\_set = set(roles)

```
async def _dep(request: Request, user: User = Depends(get_current_user)) -> User:
    if not any(r in user.roles for r in roles_set):
        _audit_log(
            decision="deny",
            reason="any_role",
            mode=_mode(),
            user=user,
            request=request,
            extra={"roles": list(roles_set), "user_roles": user.roles},
        )
        raise _AuthzError(
            status.HTTP_403_FORBIDDEN,
            {"msg": "Requires any of roles", "roles": list(roles_set), "user_roles": user.roles},
        )
    _audit_log(decision="allow", reason="any_role", mode=_mode(), user=user, request=request, extra={"ok": True})
    return user

return _dep
```

def require\_scopes(*scopes: str):
"""
Dependency factory to require ALL specified scopes (including role-inherited).
Scopes support wildcard/glob (e.g., "diagnostics:*", "write:?anifest").
"""
scopes\_list: List\[str] = list(scopes)

```
async def _dep(request: Request, user: User = Depends(get_current_user)) -> User:
    role_index = get_policy().roles
    missing = [s for s in scopes_list if not user.has_scope(s, role_index)]
    if missing:
        _audit_log(
            decision="deny",
            reason="scope",
            mode=_mode(),
            user=user,
            request=request,
            extra={"missing": missing},
        )
        raise _AuthzError(
            status.HTTP_403_FORBIDDEN,
            {"msg": "Missing required scope(s)", "missing": missing},
        )
    _audit_log(decision="allow", reason="scope", mode=_mode(), user=user, request=request, extra={"ok": True})
    return user

return _dep
```

def require\_any\_scope(\*scopes: str):
"""
Dependency factory to require ANY of the specified scopes.
"""
scopes\_list: List\[str] = list(scopes)

```
async def _dep(request: Request, user: User = Depends(get_current_user)) -> User:
    role_index = get_policy().roles
    if not user.has_any_scope(scopes_list, role_index):
        _audit_log(
            decision="deny",
            reason="any_scope",
            mode=_mode(),
            user=user,
            request=request,
            extra={"scopes": scopes_list},
        )
        raise _AuthzError(
            status.HTTP_403_FORBIDDEN,
            {"msg": "Requires any of scopes", "scopes": scopes_list},
        )
    _audit_log(decision="allow", reason="any_scope", mode=_mode(), user=user, request=request, extra={"ok": True})
    return user

return _dep
```

# -----------------------------------------------------------------------------

# Utilities for programmatic checks (non-FastAPI contexts)

# -----------------------------------------------------------------------------

def check\_roles(user: User, \*roles: str) -> None:
missing = \[r for r in roles if r not in user.roles]
if missing:
raise \_AuthzError(status.HTTP\_403\_FORBIDDEN, {"msg": "Missing roles", "missing": missing})

def check\_scopes(user: User, \*scopes: str) -> None:
role\_index = get\_policy().roles
missing = \[s for s in scopes if not user.has\_scope(s, role\_index)]
if missing:
raise \_AuthzError(status.HTTP\_403\_FORBIDDEN, {"msg": "Missing scopes", "missing": missing})

# -----------------------------------------------------------------------------

# Inline smoke test helper (optional)

# -----------------------------------------------------------------------------

def \_example\_policy() -> str:
return json.dumps(
{
"users": \[
{"id": "dev", "name": "Developer", "api\_key": "abc123", "roles": \["admin"], "scopes": \["*"]},
{"id": "viewer", "api\_key": "viewkey", "roles": \["read"], "scopes": \["diagnostics\:read", "diagnostics:*"]},
],
"roles": {
"admin": {"scopes": \["\*"]},
"read": {"scopes": \["diagnostics\:read"]},
},
},
indent=2,
)

if **name** == "**main**":
\# Minimal manual test:
\#   AUTHZ\_MODE=HEADER\_API\_KEY&#x20;
\#   AUTHZ\_API\_KEYS\_JSON='{"abc123":{"id":"dev","roles":\["admin"],"scopes":\["*"]}}'&#x20;
\#   AUTHZ\_AUDIT\_LOG="authz\_audit.jsonl"&#x20;
\#   AUTHZ\_IP\_ALLOW="127.0.0.1,::1"&#x20;
\#   AUTHZ\_RATE\_LIMIT="5/min" AUTHZ\_RATE\_BURST="5"&#x20;
\#   python -m src.server.authz
os.environ.setdefault("AUTHZ\_MODE", "HEADER\_API\_KEY")
os.environ.setdefault(
"AUTHZ\_API\_KEYS\_JSON",
'{"abc123":{"id":"dev","name":"Developer","roles":\["admin"],"scopes":\["*"]}}',
)
os.environ.setdefault("AUTHZ\_AUDIT\_LOG", "authz\_audit.jsonl")
\# Exercise cache + reload logic lightly:
p = get\_policy()
u = p.get\_by\_api\_key("abc123")
print("Loaded policy users:", list(p.users.keys()))
print("User:", u.id if u else None, "roles:", u.roles if u else None, "scopes:", u.scopes if u else None)
\# Simulate env change hot-reload:
os.environ\["AUTHZ\_DEFAULT\_ROLE"] = "read"
reset\_authz\_caches()
p2 = get\_policy()
print("Mode:", p2.\_mode, "Default role:", p2.default\_role)
