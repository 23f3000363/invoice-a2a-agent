"""Bearer token → principal isolation."""
from __future__ import annotations

import hashlib
from fastapi import Header, HTTPException, Request
from .protocol import GENERIC_UNAUTHORIZED


def principal_from_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=GENERIC_UNAUTHORIZED["error"])
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail=GENERIC_UNAUTHORIZED["error"])
    # Stable opaque principal id – never store the raw token
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def require_principal(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    return principal_from_token(authorization)
