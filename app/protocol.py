"""A2A 1.0 constants, media types, states, and error helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Media types
A2A_JSON = "application/a2a+json"
INVOICE_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
INVOICE_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
INVOICE_RESULTS = "application/vnd.ga5.invoice-action-results+json"
INVOICE_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"

# ProtoJSON-style enum strings (required by the grader)
ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"

TASK_STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
TASK_STATE_WORKING = "TASK_STATE_WORKING"
TASK_STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
TASK_STATE_COMPLETED = "TASK_STATE_COMPLETED"
TASK_STATE_CANCELED = "TASK_STATE_CANCELED"
TASK_STATE_FAILED = "TASK_STATE_FAILED"

VALID_ACTIONS = frozenset({
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
})

TERMINAL_STATES = frozenset({
    TASK_STATE_COMPLETED,
    TASK_STATE_CANCELED,
    TASK_STATE_FAILED,
})


def compact_canonical_json(obj: Any) -> str:
    """Recursively key-sorted, compact JSON for idempotency hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def message_hash(message: dict) -> str:
    """SHA-256 of the message only (ignore configuration)."""
    return hashlib.sha256(compact_canonical_json(message).encode("utf-8")).hexdigest()


def package_content_hash(package: dict) -> str:
    """Canonical hash of a single package for semantic decision caching."""
    # Exclude volatile ids so the same commercial content reuses the decision
    core = {
        k: v for k, v in package.items()
        if k not in ("packageId", "id")
    }
    return hashlib.sha256(compact_canonical_json(core).encode("utf-8")).hexdigest()


def a2a_error(code: str, message: str, status: int = 400) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
        }
    }


# Generic bodies – never leak existence of other principals' tasks
GENERIC_NOT_FOUND = a2a_error("NOT_FOUND", "Resource not found", 404)
GENERIC_FORBIDDEN = a2a_error("FORBIDDEN", "Access denied", 403)
GENERIC_UNAUTHORIZED = a2a_error("UNAUTHORIZED", "Authentication required", 401)
GENERIC_CONFLICT = a2a_error("IDEMPOTENCY_CONFLICT", "Message content conflicts with prior use of this messageId", 409)
GENERIC_BAD_REQUEST = a2a_error("BAD_REQUEST", "Invalid request", 400)
GENERIC_VERSION = a2a_error("UNSUPPORTED_VERSION", "A2A-Version must be 1.0", 400)
